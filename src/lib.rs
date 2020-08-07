// pub mod container;
mod dlpack;
// pub mod dlpack_ext;
pub mod dlpack_impl;

pub use self::dlpack::*;

#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
