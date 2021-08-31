use std::ops::{AddAssign, Mul};

#[derive(Debug, Clone)]
pub struct Matrix<T: Clone + Default> {
    width: usize,
    height: usize,
    values: Vec<T>,
}

impl<T: Clone + Default> Matrix<T> {
    pub fn new(width: usize, height: usize) -> Matrix<T> {
        let mut matrix: Matrix<T> = Matrix {
            width,
            height,
            values: Vec::with_capacity(width * height),
        };
        for _ in 0..(width * height) {
            matrix.values.push(Default::default());
        }
        matrix
    }

    pub fn init_map<F: Fn() -> T>(width: usize, height: usize, f: F) -> Matrix<T> {
        let mut matrix: Matrix<T> = Matrix {
            width,
            height,
            values: Vec::with_capacity(width * height),
        };
        for _ in 0..(width * height) {
            matrix.values.push(f());
        }
        matrix
    }

    pub fn init_random(width: usize, height: usize, range: std::ops::Range<T>) -> Matrix<T>
    where
        T: rand::distributions::uniform::SampleUniform + std::cmp::PartialOrd,
    {
        let mut matrix: Matrix<T> = Matrix {
            width,
            height,
            values: Vec::with_capacity(width * height),
        };
        use rand::prelude::*;
        let mut rng = rand::thread_rng();
        for _ in 0..(width * height) {
            matrix
                .values
                .push(rng.gen_range(range.start.clone()..range.end.clone()));
        }
        matrix
    }

    pub fn from(width: usize, height: usize, values: Vec<T>) -> Matrix<T> {
        assert_eq!(values.len(), width * height);
        let matrix: Matrix<T> = Matrix {
            width,
            height,
            values,
        };
        matrix
    }

    pub fn width(&self) -> usize {
        return self.width;
    }

    pub fn height(&self) -> usize {
        return self.height;
    }

    pub fn get(&self, x: usize, y: usize) -> Option<&T> {
        if x >= self.width || y >= self.height {
            return None;
        }
        self.values.get(y * self.width + x)
    }

    pub fn set(&mut self, x: usize, y: usize, v: T) {
        if x >= self.width || y >= self.height {
            panic!("Matrix.set: index out of bounds. x: {}, y: {}", x, y);
        }
        self.values[y * self.width + x] = v;
    }

    pub fn get_vals(self) -> Vec<T> {
        return self.values;
    }
}

impl<T: AddAssign + Mul<Output = T> + Clone + Default + Send + Sync> Mul for Matrix<T> {
    type Output = Matrix<T>;

    fn mul(self, rhs: Matrix<T>) -> Matrix<T> {
        assert_eq!(self.width, rhs.height);
        let o_w = rhs.width;
        let o_h = self.height;

        let output = crate::neural_network::parallel_worker::map_range(&(0..(o_w * o_h)), |j| {
            let x = j % o_w;
            let y = j / o_w;
            let mut val: T = Default::default();
            for i in 0..self.width {
                val += self.get(i, y).unwrap().clone() * rhs.get(x, i).unwrap().clone();
            }
            val
        })
        .unwrap();

        Matrix::from(o_w, o_h, output)
    }
}

// impl<T: AddAssign + Mul<Output = T> + Clone + Default + Send + Sync> Mul for Matrix<T> {
//     type Output = Matrix<T>;

//     fn mul(self, rhs: Matrix<T>) -> Matrix<T> {
//         assert_eq!(self.width, rhs.height);
//         let o_w = rhs.width;
//         let o_h = self.height;

//         let mut m: Matrix<T> = Matrix::new(o_w, o_h);

//         for x in 0..o_w {
//             for y in 0..o_h {
//                 let mut val: T = Default::default();
//                 for i in 0..self.width {
//                     val += self.get(i, y).unwrap().clone() * rhs.get(x, i).unwrap().clone();
//                 }
//                 m.set(x, y, val);
//             }
//         }

//         m
//     }
// }

impl<T: PartialEq + Clone + Default> PartialEq for Matrix<T> {
    fn eq(&self, other: &Matrix<T>) -> bool {
        if self.width != other.width || self.height != other.height {
            return false;
        }
        for i in 0..self.values.len() {
            if self.values[i] != other.values[i] {
                return false;
            }
        }
        true
    }
}
