use std::iter::zip;

use crate::{
    backend::cpu::{CpuBackend, CpuTensor},
    operation::{
        add_row::AddRow, cross_entropy::CrossEntropy, gate::Gate, matmul::MatMul,
        rectifier::Rectifier, sgd::Sgd, softmax::Softmax, sum_rows::SumRows,
    },
};

/// `log(0)` を避けるための微小量
const DELTA: f32 = 1e-7;

impl Rectifier for CpuBackend {
    fn rectifier(&self, x: &Self::Tensor) -> Self::Tensor {
        let x = &x.0;
        CpuTensor::new(x.data.iter().map(|&v| v.max(0.)).collect(), x.rows, x.cols)
    }
}

impl Gate for CpuBackend {
    fn gate(&self, sig: &Self::Tensor, x: &Self::Tensor) -> Self::Tensor {
        let (sig, x) = (&sig.0, &x.0);
        assert_eq!((sig.rows, sig.cols), (x.rows, x.cols));
        CpuTensor::new(
            zip(&sig.data, &x.data)
                .map(|(&s, &d)| if 0. < s { d } else { 0. })
                .collect(),
            x.rows,
            x.cols,
        )
    }
}

impl MatMul for CpuBackend {
    fn matmul(&self, a: &Self::Tensor, b: &Self::Tensor, ta: bool, tb: bool) -> Self::Tensor {
        let (a, b) = (&a.0, &b.0);
        let (m, k) = if ta {
            (a.cols, a.rows)
        } else {
            (a.rows, a.cols)
        };
        let (kb, n) = if tb {
            (b.cols, b.rows)
        } else {
            (b.rows, b.cols)
        };
        assert_eq!(k, kb);

        let mut out = vec![0.; m * n];
        for i in 0..m {
            let out_row = &mut out[i * n..(i + 1) * n];
            for p in 0..k {
                let av = if ta {
                    a.data[p * a.cols + i]
                } else {
                    a.data[i * a.cols + p]
                };
                if tb {
                    for (j, o) in out_row.iter_mut().enumerate() {
                        *o += av * b.data[j * b.cols + p];
                    }
                } else {
                    for (o, &bv) in zip(out_row.iter_mut(), &b.data[p * b.cols..(p + 1) * b.cols]) {
                        *o += av * bv;
                    }
                }
            }
        }
        CpuTensor::new(out, m, n)
    }
}

impl AddRow for CpuBackend {
    fn add_row(&self, x: &Self::Tensor, row: &Self::Tensor) -> Self::Tensor {
        let (x, row) = (&x.0, &row.0);
        assert_eq!((row.rows, row.cols), (1, x.cols));
        CpuTensor::new(
            x.data
                .iter()
                .enumerate()
                .map(|(i, &v)| v + row.data[i % x.cols])
                .collect(),
            x.rows,
            x.cols,
        )
    }
}

impl SumRows for CpuBackend {
    fn sum_rows(&self, x: &Self::Tensor) -> Self::Tensor {
        let x = &x.0;
        let mut out = vec![0.; x.cols];
        for r in 0..x.rows {
            for (o, &v) in zip(&mut out, &x.data[r * x.cols..(r + 1) * x.cols]) {
                *o += v;
            }
        }
        CpuTensor::new(out, 1, x.cols)
    }
}

impl Softmax for CpuBackend {
    fn softmax(&self, x: &Self::Tensor) -> Self::Tensor {
        let x = &x.0;
        let mut out = Vec::with_capacity(x.data.len());
        for r in 0..x.rows {
            let row = &x.data[r * x.cols..(r + 1) * x.cols];
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let start = out.len();
            out.extend(row.iter().map(|&v| (v - max).exp()));
            let sum: f32 = out[start..].iter().sum();
            out[start..].iter_mut().for_each(|v| *v /= sum);
        }
        CpuTensor::new(out, x.rows, x.cols)
    }

    fn softmax_grad(&self, y: &Self::Tensor, e: &Self::Tensor) -> Self::Tensor {
        let (y, e) = (&y.0, &e.0);
        assert_eq!((y.rows, y.cols), (e.rows, e.cols));
        let mut out = Vec::with_capacity(y.data.len());
        for r in 0..y.rows {
            let range = r * y.cols..(r + 1) * y.cols;
            let (y, e) = (&y.data[range.clone()], &e.data[range]);
            let dot: f32 = zip(y, e).map(|(&y, &e)| y * e).sum();
            out.extend(zip(y, e).map(|(&y, &e)| y * (e - dot)));
        }
        CpuTensor::new(out, y.rows, y.cols)
    }
}

impl CrossEntropy for CpuBackend {
    fn cross_entropy(&self, y: &Self::Tensor, t: &Self::Tensor) -> Self::Tensor {
        let (y, t) = (&y.0, &t.0);
        assert_eq!((y.rows, y.cols), (t.rows, t.cols));
        // t = 0 の項は 0 * ln(0) = NaN にせず 0 とみなす
        let sum: f32 = zip(&y.data, &t.data)
            .filter(|&(_, &t)| t != 0.)
            .map(|(&y, &t)| t * y.ln())
            .sum();
        CpuTensor::new(vec![-sum / y.rows as f32], 1, 1)
    }

    fn cross_entropy_grad(&self, y: &Self::Tensor, t: &Self::Tensor) -> Self::Tensor {
        let (y, t) = (&y.0, &t.0);
        assert_eq!((y.rows, y.cols), (t.rows, t.cols));
        let n = y.rows as f32;
        CpuTensor::new(
            zip(&y.data, &t.data)
                .map(|(&y, &t)| -t / (y + DELTA) / n)
                .collect(),
            y.rows,
            y.cols,
        )
    }
}

impl Sgd for CpuBackend {
    fn sgd(&self, w: &Self::Tensor, g: &Self::Tensor, lr: f32) -> Self::Tensor {
        let (w, g) = (&w.0, &g.0);
        assert_eq!((w.rows, w.cols), (g.rows, g.cols));
        CpuTensor::new(
            zip(&w.data, &g.data).map(|(&w, &g)| w - lr * g).collect(),
            w.rows,
            w.cols,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::backend::Backend;

    fn t(cpu: &CpuBackend, data: &[f32], rows: usize, cols: usize) -> CpuTensor {
        cpu.write(data, rows, cols)
    }

    fn assert_close(a: &CpuTensor, expect: &[f32], rows: usize, cols: usize) {
        assert_eq!((a.0.rows, a.0.cols), (rows, cols));
        assert_eq!(a.0.data.len(), expect.len());
        for (x, y) in zip(&a.0.data, expect) {
            assert!((x - y).abs() <= 1e-6 + 1e-5 * y.abs(), "{x} != {y}");
        }
    }

    #[test]
    fn op_gate() {
        let cpu = CpuBackend::new();
        let r = cpu.gate(
            &t(&cpu, &[-1., 0., 2., 3.], 2, 2),
            &t(&cpu, &[1., 2., 3., 4.], 2, 2),
        );
        assert_close(&r, &[0., 0., 3., 4.], 2, 2);
    }

    #[test]
    #[should_panic]
    fn op_gate_shape() {
        let cpu = CpuBackend::new();
        cpu.gate(&t(&cpu, &[1., 2.], 1, 2), &t(&cpu, &[1., 2.], 2, 1));
    }

    // a = [[1, 2, 3], [4, 5, 6]], b = [[1, 2], [3, 4], [5, 6]]
    #[test]
    fn op_matmul() {
        let cpu = CpuBackend::new();
        let a = t(&cpu, &[1., 2., 3., 4., 5., 6.], 2, 3);
        let b = t(&cpu, &[1., 2., 3., 4., 5., 6.], 3, 2);
        let expect = [22., 28., 49., 64.];
        assert_close(&cpu.matmul(&a, &b, false, false), &expect, 2, 2);

        // 転置して渡しても同じ結果になる
        let at = t(&cpu, &[1., 4., 2., 5., 3., 6.], 3, 2);
        let bt = t(&cpu, &[1., 3., 5., 2., 4., 6.], 2, 3);
        assert_close(&cpu.matmul(&at, &b, true, false), &expect, 2, 2);
        assert_close(&cpu.matmul(&a, &bt, false, true), &expect, 2, 2);
        assert_close(&cpu.matmul(&at, &bt, true, true), &expect, 2, 2);
    }

    #[test]
    #[should_panic]
    fn op_matmul_shape() {
        let cpu = CpuBackend::new();
        let a = t(&cpu, &[1., 2., 3., 4.], 2, 2);
        let b = t(&cpu, &[1., 2., 3.], 3, 1);
        cpu.matmul(&a, &b, false, false);
    }

    #[test]
    fn op_add_row() {
        let cpu = CpuBackend::new();
        let r = cpu.add_row(
            &t(&cpu, &[1., 2., 3., 4., 5., 6.], 2, 3),
            &t(&cpu, &[10., 20., 30.], 1, 3),
        );
        assert_close(&r, &[11., 22., 33., 14., 25., 36.], 2, 3);
    }

    #[test]
    fn op_sum_rows() {
        let cpu = CpuBackend::new();
        let r = cpu.sum_rows(&t(&cpu, &[1., 2., 3., 4., 5., 6.], 2, 3));
        assert_close(&r, &[5., 7., 9.], 1, 3);
    }

    #[test]
    fn op_softmax() {
        let cpu = CpuBackend::new();
        // 2行目は大きい値でもオーバーフローしない
        let r = cpu.softmax(&t(&cpu, &[0., 0., 1000., 1000.], 2, 2));
        assert_close(&r, &[0.5, 0.5, 0.5, 0.5], 2, 2);

        let r = cpu.softmax(&t(&cpu, &[1., 2., 3.], 1, 3));
        assert_close(&r, &[0.09003057, 0.24472847, 0.66524096], 1, 3);
    }

    #[test]
    fn op_softmax_grad() {
        let cpu = CpuBackend::new();
        let y = t(&cpu, &[0.25, 0.75], 1, 2);
        let e = t(&cpu, &[1., 0.], 1, 2);
        // y * (e - y.e) = [0.25 * 0.75, 0.75 * -0.25]
        assert_close(&cpu.softmax_grad(&y, &e), &[0.1875, -0.1875], 1, 2);
    }

    #[test]
    fn op_cross_entropy() {
        let cpu = CpuBackend::new();
        let y = t(&cpu, &[0.5, 0.5, 0.25, 0.75], 2, 2);
        let tt = t(&cpu, &[1., 0., 0., 1.], 2, 2);
        let expect = -(0.5f32.ln() + 0.75f32.ln()) / 2.;
        assert_close(&cpu.cross_entropy(&y, &tt), &[expect], 1, 1);
        // y = 0 でも t = 0 の項は NaN にならない
        let r = cpu.cross_entropy(&t(&cpu, &[1., 0.], 1, 2), &t(&cpu, &[1., 0.], 1, 2));
        assert_close(&r, &[0.], 1, 1);
    }

    #[test]
    fn op_cross_entropy_grad() {
        let cpu = CpuBackend::new();
        let y = t(&cpu, &[0.5, 0.5, 0.25, 0.75], 2, 2);
        let tt = t(&cpu, &[1., 0., 0., 1.], 2, 2);
        let r = cpu.cross_entropy_grad(&y, &tt);
        assert_close(&r, &[-1., 0., 0., -1. / 0.75 / 2.], 2, 2);
    }

    #[test]
    fn op_sgd() {
        let cpu = CpuBackend::new();
        let r = cpu.sgd(&t(&cpu, &[1., 2.], 1, 2), &t(&cpu, &[10., -10.], 1, 2), 0.1);
        assert_close(&r, &[0., 3.], 1, 2);
    }
}
