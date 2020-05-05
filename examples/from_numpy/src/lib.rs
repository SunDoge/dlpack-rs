// use dlpack_rs::dlpack;
// use dlpack_rs::dlpack_raw;
use dlpack_rs::dlpack::*;
// use dlpack_rs::dlpack_raw::*;
use std::ptr;

// extern "C" fn display(a: dlpack_raw::DLManagedTensor) {
//     println!("On C side:");
//     let ndim = a.dl_tensor.ndim;
// }

static mut GIVEN: *mut DLManagedTensor = ptr::null_mut();

#[no_mangle]
extern "C" fn Finalize() {
    unsafe {
        // (*given).deleter.unwrap()(given);
        // GIVEN.with(|g| (**g).deleter.unwrap()(*g))
        // (*GIVEN).deleter.unwrap()(GIVEN)
        println!("Call drop");
        println!("{:?}", (*GIVEN).dl_tensor);
        ((*GIVEN).deleter.unwrap())(GIVEN)
    };
}

#[no_mangle]
extern "C" fn Give(dl_managed_tensor: DLManagedTensor) {
    // GIVEN.with(|g| unsafe {

    //     *g = Box::into_raw(Box::new(dl_managed_tensor));
    // })
    unsafe {
        // println!("dl_managed{:?}", dl_managed_tensor);
        dbg!(dl_managed_tensor);
        GIVEN = libc::malloc(std::mem::size_of::<DLManagedTensor>()) as *mut DLManagedTensor;
        *GIVEN = dl_managed_tensor;
    };
}

#[no_mangle]
extern "C" fn hello() {
    println!("Hello from rust 👋");
}

#[no_mangle]
extern "C" fn FreeHandle() {
    unsafe {
        GIVEN.drop_in_place();
    }
}

#[no_mangle]
extern "C" fn version() {
    println!("Version: {}", DLPACK_VERSION);
}