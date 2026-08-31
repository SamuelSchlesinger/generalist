use crate::error::{Error, Result};

#[cfg(target_os = "macos")]
pub(super) fn configure_inference_thread() -> Result<()> {
    // QOS_CLASS_USER_INITIATED is Apple's public scheduling class for work the
    // user is actively waiting on. It is intentionally below USER_INTERACTIVE
    // so model execution cannot crowd out terminal input and rendering.
    const QOS_CLASS_USER_INITIATED: u32 = 0x19;

    unsafe extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
    }

    // SAFETY: the function has no pointer arguments and 0 is a valid relative
    // priority for QOS_CLASS_USER_INITIATED on supported macOS versions.
    let status = unsafe { pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0) };
    if status == 0 {
        Ok(())
    } else {
        Err(Error::Other(format!(
            "failed to set inference thread QoS: errno {status}"
        )))
    }
}

#[cfg(not(target_os = "macos"))]
pub(super) fn configure_inference_thread() -> Result<()> {
    Ok(())
}
