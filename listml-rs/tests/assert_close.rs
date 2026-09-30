mod common;
use crate::common::assert_close;

#[test]
#[should_panic]
fn assert_close_noteq_len() {
    assert_close(&[1.], &[], 0., 0.);
}

#[test]
fn assert_close_eq() {
    assert_close(&[1.], &[1.], 0., 0.);
}

#[test]
#[should_panic]
fn assert_close_noteq() {
    assert_close(&[1.], &[1.1], 0., 0.);
}

#[test]
fn assert_close_atol() {
    assert_close(&[1.], &[1.1], 0.11, 0.);
}

#[test]
fn assert_close_rtol() {
    assert_close(&[1.], &[1.1], 0.0, 0.1);
}

#[test]
#[should_panic]
fn assert_close_nottol() {
    assert_close(&[1.], &[1.2], 0.0, 0.1);
}

#[test]
#[should_panic]
fn assert_close_nan() {
    assert_close(&[f32::NAN], &[1.1], 0.1, 0.);
}

#[test]
fn assert_close_neg() {
    assert_close(&[-1.1], &[-1.], 0.0, 0.5);
}
