use std::fs;
use std::path::{Path, PathBuf};

use image::{ImageBuffer, GrayImage, Pixel, Rgba, Luma};
use imageproc::gradients::{horizontal_sobel, vertical_sobel};
use imageproc::corners::corners_fast9;
use imageproc::rect::Rect;
use imageproc::drawing::draw_hollow_rect_mut;
use nalgebra::{DMatrix, DVector, MatrixXx2};

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

fn get_feature_points(image : &GrayImage) -> Vec<(usize, usize)> {
    let corners = corners_fast9(image, 10);
    corners.into_iter().map(|c| (c.x as usize, c.y as usize)).collect()
}

/// solve the Lucas-Kanade method for a single point.
fn lucas_kanade(dx: &DMatrix<f32>, dy: &DMatrix<f32>, dt: &DMatrix<f32>, x: usize, y: usize, window_size: usize) -> Option<(f32, f32)> {
    let start = std::time::Instant::now();

    let half_window = window_size / 2;
    let mut a = Vec::new(); 
    let mut b = Vec::new();

    // check if the window exceeds matrix bounds
    if x < half_window || y < half_window || x + half_window >= dx.ncols() || y + half_window >= dx.nrows() {
        return None; // Return early if the coordinate is too close to the border
    }

    for i in (x-half_window)..=(x+half_window) {
        for j in (y-half_window)..=(y+half_window) {
            a.push([dx[(j, i)], dy[(j, i)]]);
            b.push(-dt[(j, i)]);
        }
    }

    // correct A population!
    let a = DMatrix::from_iterator(2, a.len(), a.into_iter().flatten()).transpose();
    let b = DVector::from_vec(b);

    // Compute ATA and ATb
    let ata = &a.transpose() * &a; // Results in a 2x2 matrix
    let atb = &a.transpose() * &b; // Results in a 2x1 matrix

    // Compute the inverse of ATA
    let ata_inv = ata.try_inverse().expect("Matrix is not invertible");

    // Compute the velocity vector v
    let v = ata_inv * atb; // Results in a 2x1 matrix (vector)

    let stop = start.elapsed();
    Some((v[0], v[1]))
}


// Function to iterate through images and process them
fn process_images_in_dir(dir_path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir_path)?
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            // Only include if the extension is .png
            match path.extension().and_then(|e| e.to_str()) {
                Some("png") => Some(path),
                _ => None,
            }
        })
        .collect();
    // Sort entries if necessary to ensure correct order
    entries.sort();

    let mut sum_vx: f32 = 0.0;
    let mut sum_vy: f32 = 0.0;
    let mut total_feature_points: usize = 0;

    for window in entries.windows(2) {
        let ref_path = &window[0];
        let src_path = &window[1];

        let ref_img = image::open(ref_path)?;
        let src_img = image::open(src_path)?;
        let gray_ref_img = ref_img.to_luma8();
        let gray_src_img = src_img.to_luma8();

        let (dx, dy, dt) = compute_gradients(&gray_ref_img, &gray_src_img);

        let feature_points = get_feature_points(&gray_ref_img);

        // Preparing for drawing
        // let mut ref_draw_img = ref_img.to_rgba8();
        // let mut src_draw_img = src_img.to_rgba8();
        // let color = Rgba([0u8, 255u8, 0u8, 255u8]);

        // Collect all movements and draw them at the end
        for (x, y) in feature_points {
            if let Some((vx, vy)) = lucas_kanade(&dx, &dy, &dt, x, y, 16) {
                // let ref_anchor = (x as i32, y as i32);
                // let src_anchor = ((x as f32 + vx) as i32, (y as f32 + vy) as i32);
                // draw_feature_point(&mut ref_draw_img, ref_anchor, color);
                // draw_feature_point(&mut src_draw_img, src_anchor, color);
                sum_vx += vx;
                sum_vy += vy;
                total_feature_points += 1;
            }
        }
        if total_feature_points > 0 {
            let tmp_mean_vx = sum_vx / total_feature_points as f32;
            let tmp_mean_vy = sum_vy / total_feature_points as f32;
            println!("vx: {} vy: {}", tmp_mean_vx, tmp_mean_vy);
        }
        // Save the images with drawn points
        // let _ = ref_draw_img.save("output/ref_features.png");
        // let _ = src_draw_img.save("output/src_features.png");
    }

    // if total_feature_points > 0 {
    //     mean_vx /= total_feature_points as f32;
    //     mean_vy /= total_feature_points as f32;
    //     println!("Mean vx: {}, Mean vy: {}", mean_vx, mean_vy);
    // }
    
    Ok(())
}

// Function to draw feature points
fn draw_feature_point(img: &mut ImageBuffer<Rgba<u8>, Vec<u8>>, anchor: (i32, i32), color: Rgba<u8>) {
    let rect = Rect::at(anchor.0 - 2, anchor.1 - 2).of_size(4, 4);
    draw_hollow_rect_mut(img, rect, color);
}

fn main() {
    let dir_path = Path::new("/Users/thomas/git/personal/calib_challenge/images/1/"); // Change this to your directory path
    if let Err(e) = process_images_in_dir(dir_path) {
        eprintln!("Error processing images: {}", e);
    }
}


fn create_synthetic_image(width: u32, height: u32, dot_position: (u32, u32)) -> GrayImage {
    let mut img = GrayImage::new(width, height);
    img.put_pixel(dot_position.0, dot_position.1, Luma([255u8]));
    img
}

fn test_synthetic() {
    let width = 10;
    let height = 10;

    // Create two synthetic images
    let dot_initial_position = (5, 5);
    let dot_offset_position = (5, 6); // Offset by 5 pixels in both x and y directions

    let img1 = create_synthetic_image(width, height, dot_initial_position);
    let img2 = create_synthetic_image(width, height, dot_offset_position);

    // Now integrate this with your existing Lucas-Kanade code to calculate the displacement
    // Assume compute_gradients, lucas_kanade, etc., are defined as in your initial code snippet

    let (dx, dy, dt) = compute_gradients(&img1, &img2);

    // Assuming the Lucas-Kanade is applied at the initial dot position
    // You might want to ensure the dot is within a region that the window size does not exceed image bounds
    if let Some((vx, vy)) = lucas_kanade(&dx, &dy, &dt, dot_initial_position.0 as usize, dot_initial_position.1 as usize, 4) {
        println!("Calculated displacement: vx = {}, vy = {}", vx, vy);
        // Expected output: vx = 5, vy = 5 (or close to these values depending on the implementation)
    } 
}
