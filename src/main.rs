use std::error;
use image::{GrayImage, Pixel, ImageBuffer, Luma};
use imageproc::gradients::{horizontal_sobel, vertical_sobel};
use nalgebra::{DMatrix, DVector, Matrix2xX};
use nalgebra::linalg::SVD;

/// helper function to convert a GrayImage to a nalgebra DMatrix<f32>
// fn image_to_matrix(image: &GrayImage) -> DMatrix<f32> {
//     let (width, height) = image.dimensions();
//     let mut matrix = DMatrix::zeros(height as usize, width as usize);
//     for y in 0..height {
//         for x in 0..width {
//             let pixel = image.get_pixel(x, y).0[0] as f32;
//             matrix[(y as usize, x as usize)] = pixel;
//         }
//     }
//     matrix
// 

fn image_to_matrix<P, Container>(image: &image::ImageBuffer<P, Container>) -> DMatrix<f32>
where
    P: Pixel + 'static,
    P::Subpixel: Into<f32>,
    Container: std::ops::Deref<Target = [P::Subpixel]>,
{
    let (width, height) = image.dimensions();
    let mut matrix = DMatrix::zeros(height as usize, width as usize);
    for y in 0..height {
        for x in 0..width {
            let pixel = image.get_pixel(x, y).to_luma().0[0].into();
            matrix[(y as usize, x as usize)] = pixel as f32;
        }
    }
    matrix
}


/// compute the spatial gradients (dx, dy) and temporal gradient (dt) between two images.
fn compute_gradients(current: &GrayImage, next: &GrayImage) -> (DMatrix<f32>, DMatrix<f32>, DMatrix<f32>) {

    // convert images to nalgebra for element-wise operations.
    let current_matrix = image_to_matrix(current);
    let next_matrix = image_to_matrix(next);

    // calculate temporal gradient
    let dt = &next_matrix - &current_matrix;

    // use imageproc's gradients_sobel for spatial gradients
    let dx = horizontal_sobel(current);
    let dy = vertical_sobel(current);

    // convert gradients to nalgebra matrices
    let dx_matrix = image_to_matrix(&dx);
    let dy_matrix = image_to_matrix(&dy);

    (dx_matrix, dy_matrix, dt)
}

/// Solve the Lucas-Kanade method for a single point.
fn lucas_kanade(dx: &DMatrix<f32>, dy: &DMatrix<f32>, dt: &DMatrix<f32>, x: usize, y: usize, window_size: usize) -> Option<(f32, f32)> {
    let half_window = window_size / 2;
    
    let mut a = Vec::new();
    let mut b = Vec::new();

    // Construct A and b matrices from gradients within the window
    for i in (x-half_window)..=(x+half_window) {
        for j in (y-half_window)..=(y+half_window) {
            a.push([dx[(j, i)], dy[(j, i)]]);
            b.push(-dt[(j, i)]);
        }
    }

    let a = Matrix2xX::from_iterator(a.len(), a.into_iter().flatten());
    let b = DVector::from_vec(b);

    println!("{}x{} {}x{}", a.nrows(), a.ncols(), b.nrows(), b.ncols());

    let ata = a.clone().transpose() * a.clone(); // This should be [2, 2]
    println!("OKOKOK");
    let atb = a.clone().transpose() * b.clone(); // Make sure this is [2, 1]
      
    println!("{}x{} {}x{}", ata.nrows(), ata.ncols(), atb.nrows(), atb.ncols());

    // Solve using SVD to handle potential singularity in A^TA
    let svd = SVD::new(a.clone(), true, true);
    let solution = svd.solve(&atb, 1e-6); // 1e-6 is the threshold for singular values
    // println!("WE ARE HERE");

    solution.map(|v| (v[0], v[1])).ok()
}

fn load_images() -> Result<(GrayImage, GrayImage), Box<dyn error::Error>> {
    let ref_path = "/Users/thomas/git/personal/calib_challenge/images/0/000001.png";
    let src_path = "/Users/thomas/git/personal/calib_challenge/images/0/000002.png";
    let ref_img = image::open(ref_path)?;
    let src_img = image::open(src_path)?;
    let gray_ref_img = ref_img.grayscale().to_luma8();
    let gray_src_img = src_img.grayscale().to_luma8();

    Ok((gray_ref_img, gray_src_img))

}

fn main() {
    // load reference and current image
    let (gray_ref_img, gray_src_img) = load_images().expect("Failed to load images");

    // compute image grads
    let (dy, dx, dt) = compute_gradients(&gray_ref_img, &gray_src_img);

    // compute LKT
    let lkt = lucas_kanade(&dx, &dy, &dt, 100, 100, 16).unwrap();

    println!("LKT :: {} {}\n", lkt.0, lkt.1);
}
