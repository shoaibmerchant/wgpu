use glow::HasContext;
use glutin::{
    api::egl::Egl,
    config::{Api, Config, ConfigSurfaceTypes, GlConfig},
    context::{AsRawContext, ContextApi, ContextAttributesBuilder, RawContext, Version},
    display::{AsRawDisplay, DisplayApiPreference, GetDisplayExtensions, GetGlDisplay},
    prelude::{GlDisplay, NotCurrentGlContext, PossiblyCurrentGlContext},
    surface::{AsRawSurface, GlSurface, WindowSurface},
};
use khronos_egl::Downcast;
use once_cell::sync::Lazy;
use parking_lot::{MappedMutexGuard, Mutex, MutexGuard, RwLock};
use raw_window_handle::WaylandDisplayHandle;

use std::{
    collections::HashMap,
    ffi::{self, CString},
    mem::ManuallyDrop,
    num::NonZero,
    ops::DerefMut,
    os::raw,
    ptr,
    rc::Rc,
    sync::Arc,
    time::Duration,
};

use crate::InstanceError;

/// The amount of time to wait while trying to obtain a lock to the adapter context
const CONTEXT_LOCK_TIMEOUT_SECS: u64 = 1;

fn parse_egl_version(version_str: &str) -> Option<(i32, i32)> {
    // Expects format: "EGL {major}.{minor}"
    let parts: Vec<&str> = version_str.trim().split_whitespace().collect();
    if parts.len() != 2 {
        return None;
    }
    let version_parts: Vec<&str> = parts[1].split('.').collect();
    if version_parts.len() != 2 {
        return None;
    }
    let major = version_parts[0].parse().ok()?;
    let minor = version_parts[1].parse().ok()?;
    Some((major, minor))
}

#[derive(Clone, Debug)]
pub struct EglContext {
    pub context: Arc<glutin::api::egl::context::PossiblyCurrentContext>,
    version: (i32, i32),
    pub display: glutin::api::egl::display::Display,
    // pbuffer: Option<Arc<glutin::api::egl::surface::Surface<glutin::surface::PbufferSurface>>,
}

impl EglContext {
    pub fn make_current(&self) -> Result<(), glutin::error::Error> {
        self.context.make_current_surfaceless()
    }

    pub fn unmake_current(&self) -> Result<(), glutin::error::Error> {
        // TODO is make_not_current_in_place() okay, or should we switch to make_not_current?
        self.context.make_not_current_in_place()
    }
}

/// A wrapper around a [`glow::Context`] and the required EGL context that uses locking to guarantee
/// exclusive access when shared with multiple threads.
pub struct AdapterContext {
    glow: Mutex<ManuallyDrop<glow::Context>>,
    egl: Option<EglContext>,
}

unsafe impl Sync for AdapterContext {}
unsafe impl Send for AdapterContext {}

impl AdapterContext {
    pub fn is_owned(&self) -> bool {
        self.egl.is_some()
    }

    /// Returns the EGL instance.
    ///
    /// This provides access to EGL functions and the ability to load GL and EGL extension functions.
    pub fn egl_instance(&self) -> Option<&Egl> {
        self.egl.as_ref().map(|egl| &*egl.display.egl())
    }

    /// Returns the EGLDisplay corresponding to the adapter context.
    ///
    /// Returns [`None`] if the adapter was externally created.
    pub fn raw_display(&self) -> Option<glutin::api::egl::display::Display> {
        let display = match self.egl {
            Some(ref egl) => Some(egl.display.clone()),
            None => None,
        };
        display
    }

    /// Returns the EGL version the adapter context was created with.
    ///
    /// Returns [`None`] if the adapter was externally created.
    pub fn egl_version(&self) -> Option<(i32, i32)> {
        let version = match self.egl {
            Some(ref egl) => Some(egl.version),
            None => None,
        };
        version
    }

    pub fn raw_context(&self) -> Option<RawContext> {
        match self.egl {
            Some(ref egl) => Some(egl.context.raw_context()),
            None => None,
        }
    }
}

impl Drop for AdapterContext {
    fn drop(&mut self) {
        struct CurrentGuard<'a>(&'a EglContext);
        impl Drop for CurrentGuard<'_> {
            fn drop(&mut self) {
                let res = self.0.unmake_current();
                log::error!(
                    "GLUTIN_DEBUG: AdapterContext::drop unmake_current() - {:?}",
                    res
                );
            }
        }

        // Context must be current when dropped. See safety docs on
        // `glow::HasContext`.
        //
        // NOTE: This is only set to `None` by `Adapter::new_external` which
        // requires the context to be current when anything that may be holding
        // the `Arc<AdapterShared>` is dropped.
        let _guard = self.egl.as_ref().map(|egl| {
            let res = egl.make_current();
            log::error!(
                "GLUTIN_DEBUG: AdapterContext::drop make_current() - {:?}",
                res
            );
            CurrentGuard(&egl)
        });
        let glow = self.glow.get_mut();
        // SAFETY: Field not used after this.
        unsafe { ManuallyDrop::drop(glow) };
    }
}

struct EglContextLock<'a> {
    context: &'a Arc<glutin::api::egl::context::PossiblyCurrentContext>,
    display: glutin::api::egl::display::Display,
}

/// A guard containing a lock to an [`AdapterContext`], while the GL context is kept current.
pub struct AdapterContextLock<'a> {
    glow: MutexGuard<'a, ManuallyDrop<glow::Context>>,
    egl: Option<EglContextLock<'a>>,
}

impl<'a> std::ops::Deref for AdapterContextLock<'a> {
    type Target = glow::Context;

    fn deref(&self) -> &Self::Target {
        &self.glow
    }
}

impl<'a> Drop for AdapterContextLock<'a> {
    fn drop(&mut self) {
        if let Some(egl) = self.egl.take() {
            let res = egl.context.make_not_current_in_place();
            if res.is_err() {
                log::error!("Cannot make_not_current_in_place() - {:?}", res.err());
            }
        }
    }
}

impl AdapterContext {
    /// Get's the [`glow::Context`] without waiting for a lock
    ///
    /// # Safety
    ///
    /// This should only be called when you have manually made sure that the current thread has made
    /// the EGL context current and that no other thread also has the EGL context current.
    /// Additionally, you must manually make the EGL context **not** current after you are done with
    /// it, so that future calls to `lock()` will not fail.
    ///
    /// > **Note:** Calling this function **will** still lock the [`glow::Context`] which adds an
    /// > extra safe-guard against accidental concurrent access to the context.
    pub unsafe fn get_without_egl_lock(&self) -> MappedMutexGuard<glow::Context> {
        let guard = self
            .glow
            .try_lock_for(Duration::from_secs(CONTEXT_LOCK_TIMEOUT_SECS))
            .expect("Could not lock adapter context. This is most-likely a deadlock.");
        MutexGuard::map(guard, |glow| &mut **glow)
    }

    /// Obtain a lock to the EGL context and get handle to the [`glow::Context`] that can be used to
    /// do rendering.
    #[track_caller]
    pub fn lock<'a>(&'a self) -> AdapterContextLock<'a> {
        let glow = self
            .glow
            // Don't lock forever. If it takes longer than 1 second to get the lock we've got a
            // deadlock and should panic to show where we got stuck
            .try_lock_for(Duration::from_secs(CONTEXT_LOCK_TIMEOUT_SECS))
            .expect("Could not lock adapter context. This is most-likely a deadlock.");

        let egl = self.egl.as_ref().map(|egl| {
            let res = egl.make_current();
            log::error!(
                "GLUTIN_DEBUG: AdapterContext::lock make_current() - {:?}",
                res
            );
            EglContextLock {
                context: &egl.context,
                display: egl.display.clone(),
            }
        });

        AdapterContextLock { glow, egl }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum SrgbFrameBufferKind {
    /// No support for SRGB surface
    None,
    /// Using EGL 1.5's support for colorspaces
    Core,
    /// Using EGL_KHR_gl_colorspace
    Khr,
}

// Choose GLES framebuffer configuration
fn choose_config(
    display: glutin::api::egl::display::Display,
    srgb_kind: SrgbFrameBufferKind,
) -> Result<(glutin::api::egl::config::Config, bool), crate::InstanceError> {
    // // first try to find a pbuffer display
    // let mut pbuffer_template = glutin::config::ConfigTemplateBuilder::new()
    //     .with_surface_type(ConfigSurfaceTypes::PBUFFER)
    //     .with_api(Api::GLES2);
    // if srgb_kind != SrgbFrameBufferKind::None {
    //     pbuffer_template = pbuffer_template.with_alpha_size(8);
    // }

    // let config = unsafe {
    //     display
    //         .find_configs(pbuffer_template.build())
    //         .unwrap()
    //         .next()
    //         // .ok_or(crate::InstanceError::new("No EGL config found supporting a ".to_owned()))?
    // };

    // if config.is_some() {
    //     return Ok((config.unwrap(), false));
    // }

    // now search for a config that supports window surface type
    let mut window_template =
        glutin::config::ConfigTemplateBuilder::new().with_surface_type(ConfigSurfaceTypes::WINDOW);
    if srgb_kind != SrgbFrameBufferKind::None {
        window_template = window_template.with_alpha_size(8);
    }

    // let config = unsafe {
    //     display
    //         .find_configs(window_template.build())
    //         .unwrap()
    //         .next()
    //         .ok_or(InstanceError::new("No EGL config found for window surface".to_owned()))?
    // };

    let configs = unsafe {
        display
            .find_configs(window_template.build())
            .ok()
            .expect("No EGL config found for window surface")
    };

    let config = configs.reduce(|accum, config| {
        let transparency_check = config.supports_transparency().unwrap_or(false)
            & !accum.supports_transparency().unwrap_or(false);

        if transparency_check || config.num_samples() > accum.num_samples() {
            config
        } else {
            accum
        }
    });

    match config {
        Some(c) => Ok((c, true)),
        None => Err(InstanceError::new(
            "No EGL config found for window surface".to_owned(),
        )),
    }
}

#[derive(Debug)]
struct Inner {
    /// Note: the context contains a dummy pbuffer (1x1).
    /// Required for `eglMakeCurrent` on platforms that doesn't supports `EGL_KHR_surfaceless_context`.
    egl: EglContext,
    #[allow(unused)]
    version: (i32, i32),
    supports_native_window: bool,
    config: glutin::api::egl::config::Config,
    // #[cfg_attr(Emscripten, allow(dead_code))]
    wl_display: Option<*mut raw::c_void>,
    #[cfg_attr(Emscripten, allow(dead_code))]
    force_gles_minor_version: wgt::Gles3MinorVersion,
    /// Method by which the framebuffer should support srgb
    srgb_kind: SrgbFrameBufferKind,
}

impl Inner {
    fn create(
        flags: wgt::InstanceFlags,
        display: glutin::api::egl::display::Display,
        force_gles_minor_version: wgt::Gles3MinorVersion,
    ) -> Result<Self, crate::InstanceError> {
        profiling::scope!("Create EGL Context");

        let display_version = display.version_string();
        let version = parse_egl_version(&display_version).expect("EGL version cannot be parsed");
        let display_extensions = display.extensions();

        log::info!("Display version: {:?}", display_version);
        log::info!("Display extensions: {:?}", display_extensions);

        let srgb_kind = if version >= (1, 5) {
            log::debug!("\tEGL surface: +srgb");
            SrgbFrameBufferKind::Core
        } else if display_extensions.contains("EGL_KHR_gl_colorspace") {
            log::debug!("\tEGL surface: +srgb khr");
            SrgbFrameBufferKind::Khr
        } else {
            log::warn!("\tEGL surface: -srgb");
            SrgbFrameBufferKind::None
        };

        let (config, supports_native_window) = match choose_config(display.clone(), srgb_kind) {
            Ok((c, n)) => (c, n),
            Err(e) => {
                return Err(InstanceError::new(
                    "No matching gl config found".to_string(),
                ));
            }
        };

        log::debug!("Config {:?}", config);
        log::debug!("Config color_buffer_type {:?}", config.color_buffer_type());
        log::debug!("Config float_pixels {:?}", config.float_pixels());
        log::debug!("Config alpha_size {:?}", config.alpha_size());
        log::debug!("Config srgb_capable {:?}", config.srgb_capable());
        log::debug!("Config depth_size {:?}", config.depth_size());
        log::debug!("Config stencil_size {:?}", config.stencil_size());
        log::debug!("Config num_samples {:?}", config.num_samples());
        log::debug!(
            "Config config_surface_types {:?}",
            config.config_surface_types()
        );
        log::debug!(
            "Config hardware_accelerated {:?}",
            config.hardware_accelerated()
        );
        log::debug!(
            "Config supports_transparency {:?}",
            config.supports_transparency()
        );
        log::debug!("Config api {:?}", config.api());

        let context_attributes = ContextAttributesBuilder::new().build(None);

        // Since glutin by default tries to create OpenGL core context, which may not be
        // present we should try gles.
        let fallback_context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::Gles(None))
            .build(None);

        // There are also some old devices that support neither modern OpenGL nor GLES.
        // To support these we can try and create a 2.1 context.
        let legacy_context_attributes = ContextAttributesBuilder::new()
            .with_context_api(ContextApi::OpenGl(Some(Version::new(2, 1))))
            .build(None);

        // TODO port from gles handle robustness, opengl / opengles

        let context: glutin::api::egl::context::NotCurrentContext = unsafe {
            display
                .create_context(&config, &context_attributes)
                .unwrap_or_else(|_| {
                    display
                        .create_context(&config, &fallback_context_attributes)
                        .unwrap_or_else(|_| {
                            display
                                .create_context(&config, &legacy_context_attributes)
                                .expect("failed to create context")
                        })
                })
        };

        // Create a dummy pbuffer surface
        // TODO: Testing if context can be binded without surface
        // and creating dummy pbuffer surface if not.

        // let pbuffer = if version >= (1, 5)
        //     || display_extensions.contains("EGL_KHR_surfaceless_context")
        //     || cfg!(Emscripten)
        // {
        //     log::debug!("\tEGL context: +surfaceless");
        //     None
        // } else {
        //     let attributes = [
        //         khronos_egl::WIDTH,
        //         1,
        //         khronos_egl::HEIGHT,
        //         1,
        //         khronos_egl::NONE,
        //     ];
        //     egl.create_pbuffer_surface(display, config, &attributes)
        //         .map(Some)
        //         .map_err(|e| {
        //             crate::InstanceError::with_source(
        //                 String::from("error in create_pbuffer_surface"),
        //                 e,
        //             )
        //         })?
        // };

        // Make context current
        // let context = not_current_gl_context.make_current(&pbuffer).unwrap();

        Ok(Self {
            egl: EglContext {
                display,
                context: Arc::new(context.treat_as_possibly_current()),
                // pbuffer: None,
                version,
            },
            wl_display: None,
            version: (3, 0), // Example version
            supports_native_window,
            config,
            force_gles_minor_version,
            srgb_kind,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum WindowKind {
    Wayland,
    X11,
    AngleX11,
    Unknown,
}

#[derive(Clone, Debug)]
struct WindowSystemInterface {
    kind: WindowKind,
}

pub struct Instance {
    wsi: WindowSystemInterface,
    flags: wgt::InstanceFlags,
    inner: Mutex<Inner>,
}

impl Instance {}

unsafe impl Send for Instance {}
unsafe impl Sync for Instance {}

impl crate::Instance for Instance {
    type A = super::Api;

    unsafe fn init(desc: &crate::InstanceDescriptor) -> Result<Self, crate::InstanceError> {
        profiling::scope!("Init OpenGL (EGL) Backend");

        let display = unsafe {
            glutin::api::egl::display::Display::new(
                desc.display
                    .expect("cannot create glutin instance without raw display handle")
                    .as_raw(),
            )
            .expect("couldn't create glutin display")
        };

        // TODO: print client extensions
        // TODO: Port gles backend debug code - context attributes builder has with_debug()

        let inner = Inner::create(desc.flags, display, desc.gles_minor_version)?;

        Ok(Instance {
            wsi: WindowSystemInterface {
                kind: WindowKind::Wayland,
            },
            flags: desc.flags,
            inner: Mutex::new(inner),
        })
    }

    unsafe fn create_surface(
        &self,
        display_handle: raw_window_handle::RawDisplayHandle,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Result<<Self::A as crate::Api>::Surface, InstanceError> {
        let inner = self.inner.lock();

        // // create new inner that uses the display sent in instance's desc
        // let new_inner = Inner::create(
        //     self.flags,
        //     self.display.clone(),
        //     inner.force_gles_minor_version,
        // ).unwrap();

        // let old_inner = std::mem::replace(inner.deref_mut(), new_inner);

        // // let wl_display_handle = WaylandDisplayHandle::new(display_handle);
        // inner.wl_display = Some(wl_display_handle.display.as_ptr());

        // drop(old_inner);

        let res = inner.egl.unmake_current();
        log::error!(
            "GLUTIN_DEBUG: Instance::create_surface unmake_current() - {:?}",
            res
        );

        Ok(Surface {
            egl: inner.egl.clone(),
            wsi: self.wsi.clone(),
            config: inner.config.clone(),
            presentable: inner.supports_native_window,
            raw_window_handle: window_handle,
            swapchain: RwLock::new(None),
            srgb_kind: inner.srgb_kind,
        })
    }

    unsafe fn enumerate_adapters(
        &self,
        _surface_hint: Option<&<Self::A as crate::Api>::Surface>,
    ) -> Vec<crate::ExposedAdapter<Self::A>> {
        let inner = self.inner.lock();

        let res = inner.egl.make_current();
        log::error!(
            "GLUTIN_DEBUG: Instance::enumerate_adapters make_current() - {:?}",
            res
        );

        let mut gl = unsafe {
            glow::Context::from_loader_function(|name| {
                inner
                    .egl
                    .display
                    .get_proc_address(CString::new(name).unwrap().as_c_str())
            })
        };

        // In contrast to OpenGL ES, OpenGL requires explicitly enabling sRGB conversions,
        // as otherwise the user has to do the sRGB conversion.
        if !matches!(inner.srgb_kind, SrgbFrameBufferKind::None) {
            unsafe { gl.enable(glow::FRAMEBUFFER_SRGB) };
        }

        if self.flags.contains(wgt::InstanceFlags::DEBUG) && gl.supports_debug() {
            log::debug!("Max label length: {}", unsafe {
                gl.get_parameter_i32(glow::MAX_LABEL_LENGTH)
            });
        }

        if self.flags.contains(wgt::InstanceFlags::VALIDATION) && gl.supports_debug() {
            log::debug!("Enabling GLES debug output");
            unsafe { gl.enable(glow::DEBUG_OUTPUT) };
            unsafe { gl.debug_message_callback(super::gl_debug_message_callback) };
        }

        // Wrap in ManuallyDrop to make it easier to "current" the GL context before dropping this
        // GLOW context, which could also happen if a panic occurs after we uncurrent the context
        // below but before AdapterContext is constructed.
        let gl = ManuallyDrop::new(gl);

        let res = inner.egl.unmake_current();
        log::error!(
            "GLUTIN_DEBUG: Instance::enumerate_adapters unmake_current() - {:?}",
            res
        );

        unsafe {
            super::Adapter::expose(AdapterContext {
                glow: Mutex::new(gl),
                egl: Some(inner.egl.clone()),
            })
        }
        .into_iter()
        .collect()
    }
}

#[derive(Debug)]
pub struct Swapchain {
    surface: glutin::api::egl::surface::Surface<WindowSurface>,
    // TODO: remove the wl_window as we are having the raw_window_handle(somewhere)
    // wl_window: Option<*mut raw::c_void>,
    framebuffer: glow::Framebuffer,
    renderbuffer: glow::Renderbuffer,
    /// Extent because the window lies
    extent: wgt::Extent3d,
    format: wgt::TextureFormat,
    format_desc: super::TextureFormatDesc,
    #[allow(unused)]
    sample_type: wgt::TextureSampleType,
}
#[derive(Debug)]
pub struct Surface {
    egl: EglContext,
    wsi: WindowSystemInterface,
    config: glutin::api::egl::config::Config,
    pub(super) presentable: bool,
    raw_window_handle: raw_window_handle::RawWindowHandle,
    swapchain: RwLock<Option<Swapchain>>,
    srgb_kind: SrgbFrameBufferKind,
}

unsafe impl Send for Surface {}
unsafe impl Sync for Surface {}

impl Surface {
    pub(super) unsafe fn present(
        &self,
        _suf_texture: super::Texture,
        context: &AdapterContext,
    ) -> Result<(), crate::SurfaceError> {
        let gl = unsafe { context.get_without_egl_lock() };
        let swapchain = self.swapchain.read();
        let sc = swapchain.as_ref().unwrap();

        let res = self.egl.context.make_current(&sc.surface);
        log::error!(
            "GLUTIN_DEBUG: Surface::present() Failed make_current() - {:?}",
            res
        );

        unsafe { gl.disable(glow::SCISSOR_TEST) };
        unsafe { gl.color_mask(true, true, true, true) };

        unsafe { gl.bind_framebuffer(glow::DRAW_FRAMEBUFFER, None) };
        unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(sc.framebuffer)) };

        if !matches!(self.srgb_kind, SrgbFrameBufferKind::None) {
            // Disable sRGB conversions for `glBlitFramebuffer` as behavior does diverge between
            // drivers and formats otherwise and we want to ensure no sRGB conversions happen.
            unsafe { gl.disable(glow::FRAMEBUFFER_SRGB) };
        }

        // Note the Y-flipping here. GL's presentation is not flipped,
        // but main rendering is. Therefore, we Y-flip the output positions
        // in the shader, and also this blit.
        unsafe {
            gl.blit_framebuffer(
                0,
                sc.extent.height as i32,
                sc.extent.width as i32,
                0,
                0,
                0,
                sc.extent.width as i32,
                sc.extent.height as i32,
                glow::COLOR_BUFFER_BIT,
                glow::NEAREST,
            )
        };

        if !matches!(self.srgb_kind, SrgbFrameBufferKind::None) {
            unsafe { gl.enable(glow::FRAMEBUFFER_SRGB) };
        }

        unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None) };

        let _ = sc.surface.swap_buffers(&self.egl.context);

        // make current surfaceless
        let res = self.egl.unmake_current();
        log::error!(
            "GLUTIN_DEBUG: Surface::present unmake_current() - {:?}",
            res
        );

        Ok(())
    }

    unsafe fn unconfigure_impl(
        &self,
        device: &super::Device,
    ) -> Option<glutin::api::egl::surface::Surface<WindowSurface>> {
        let gl = &device.shared.context.lock();
        match self.swapchain.write().take() {
            Some(sc) => {
                unsafe { gl.delete_renderbuffer(sc.renderbuffer) };
                unsafe { gl.delete_framebuffer(sc.framebuffer) };
                Some(sc.surface)
            }
            None => None,
        }
    }

    pub fn supports_srgb(&self) -> bool {
        match self.srgb_kind {
            SrgbFrameBufferKind::None => false,
            _ => true,
        }
    }
}

impl crate::Surface for Surface {
    type A = super::Api;

    unsafe fn configure(
        &self,
        device: &super::Device,
        config: &crate::SurfaceConfiguration,
    ) -> Result<(), crate::SurfaceError> {
        let surface = match unsafe { self.unconfigure_impl(device) } {
            Some(pair) => pair,
            None => {
                let attributes_builder =
                    glutin::surface::SurfaceAttributesBuilder::<WindowSurface>::new();
                // We don't want any of the buffering done by the driver, because we
                // manage a swapchain on our side.
                // Some drivers just fail on surface creation seeing `EGL_SINGLE_BUFFER`.
                // if cfg!(any(target_os = "android", target_os = "macos"))
                //     || cfg!(windows)
                //     || self.wsi.kind == WindowKind::AngleX11
                // {
                //     khronos_egl::BACK_BUFFER
                // } else {
                //     khronos_egl::SINGLE_BUFFER
                // },
                let attributes_builder = attributes_builder.with_single_buffer(false);
                let attributes_builder = if config.format.is_srgb() {
                    match self.srgb_kind {
                        SrgbFrameBufferKind::None => attributes_builder.with_srgb(None),
                        _ => attributes_builder.with_srgb(Some(true)),
                    }
                } else {
                    attributes_builder
                };

                let attributes = attributes_builder.build(
                    self.raw_window_handle,
                    NonZero::new(config.extent.width)
                        .expect("trying to configure the surface with a negative or zero width"),
                    NonZero::new(config.extent.height)
                        .expect("trying to configure the surface with a negative or zero height"),
                );
                let surface = unsafe {
                    self.egl
                        .display
                        .create_window_surface(&self.config, &attributes)
                        .expect("couldn't create surface")
                };
                surface
            }
        };

        // if let Some(window) = wl_window {
        //     let library = &self.wsi.display_owner.as_ref().unwrap().library;
        //     let wl_egl_window_resize: libloading::Symbol<WlEglWindowResizeFun> =
        //         unsafe { library.get(b"wl_egl_window_resize\0") }.unwrap();
        //     unsafe {
        //         wl_egl_window_resize(
        //             window,
        //             config.extent.width as i32,
        //             config.extent.height as i32,
        //             0,
        //             0,
        //         )
        //     };
        // }

        let format_desc = device.shared.describe_texture_format(config.format);
        let gl = &device.shared.context.lock();
        let renderbuffer = unsafe { gl.create_renderbuffer() }.map_err(|error| {
            log::error!("Internal swapchain renderbuffer creation failed: {error}");
            crate::DeviceError::OutOfMemory
        })?;
        unsafe { gl.bind_renderbuffer(glow::RENDERBUFFER, Some(renderbuffer)) };
        unsafe {
            gl.renderbuffer_storage(
                glow::RENDERBUFFER,
                format_desc.internal,
                config.extent.width as _,
                config.extent.height as _,
            )
        };
        let framebuffer = unsafe { gl.create_framebuffer() }.map_err(|error| {
            log::error!("Internal swapchain framebuffer creation failed: {error}");
            crate::DeviceError::OutOfMemory
        })?;
        unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, Some(framebuffer)) };
        unsafe {
            gl.framebuffer_renderbuffer(
                glow::READ_FRAMEBUFFER,
                glow::COLOR_ATTACHMENT0,
                glow::RENDERBUFFER,
                Some(renderbuffer),
            )
        };
        unsafe { gl.bind_renderbuffer(glow::RENDERBUFFER, None) };
        unsafe { gl.bind_framebuffer(glow::READ_FRAMEBUFFER, None) };

        let mut swapchain = self.swapchain.write();
        *swapchain = Some(Swapchain {
            surface,
            renderbuffer,
            framebuffer,
            extent: config.extent,
            format: config.format,
            format_desc,
            sample_type: wgt::TextureSampleType::Float { filterable: false },
        });

        Ok(())
    }

    unsafe fn unconfigure(&self, device: &super::Device) {
        if let Some((surface)) = unsafe { self.unconfigure_impl(device) } {
            drop(surface);
        }
    }

    unsafe fn acquire_texture(
        &self,
        _timeout: Option<Duration>,
        _fence: &<Self::A as crate::Api>::Fence,
    ) -> Result<Option<crate::AcquiredSurfaceTexture<Self::A>>, crate::SurfaceError> {
        let swapchain = self.swapchain.read();
        let sc = swapchain.as_ref().unwrap();
        let texture = super::Texture {
            inner: super::TextureInner::Renderbuffer {
                raw: sc.renderbuffer,
            },
            drop_guard: None,
            array_layer_count: 1,
            mip_level_count: 1,
            format: sc.format,
            format_desc: sc.format_desc.clone(),
            copy_size: crate::CopyExtent {
                width: sc.extent.width,
                height: sc.extent.height,
                depth: 1,
            },
        };
        Ok(Some(crate::AcquiredSurfaceTexture {
            texture,
            suboptimal: false,
        }))
    }

    unsafe fn discard_texture(&self, _texture: <Self::A as crate::Api>::SurfaceTexture) {}
}
