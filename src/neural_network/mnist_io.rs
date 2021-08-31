use std::thread;
use std::{io::Read, io::Seek};
extern crate num_cpus;

use crate::neural_network::neural_network::NetworkValueType;

pub struct Image {
    pub pixels: Vec<NetworkValueType>,
    pub label: u8,
}

#[derive(Copy, Clone)]
struct ImageDataSetInfo {
    magic_num: u32,
    num_images: u32,
    image_width: u32,
    image_height: u32,
}

#[derive(Copy, Clone)]
struct LabelDataSetInfo {
    magic_num: u32,
    num_images: u32,
}

pub fn load_image_set(image_path: &str, label_path: &str) -> Result<Vec<Image>, String> {
    let mut image_file_metadata: [u8; 16] = [0; 16];
    read_file_part_into_buffer(image_path, 0, &mut image_file_metadata);

    let mut label_file_metadata: [u8; 8] = [0; 8];
    read_file_part_into_buffer(label_path, 0, &mut label_file_metadata);

    let image_set_info = load_image_set_info(&image_file_metadata);
    let label_set_info = load_label_set_info(&label_file_metadata);

    if image_set_info.magic_num != 2051 {
        return Err(format!("Invalid Image Set File (Magic number not 2051)"));
    }

    if label_set_info.magic_num != 2049 {
        return Err(format!("Invalid Label Set File (Magic number not 2049)"));
    }

    if image_set_info.num_images != label_set_info.num_images {
        return Err(format!(
            "Image and Label set lengths are not equal: {} != {}",
            image_set_info.num_images, label_set_info.num_images
        ));
    }

    let num_threads = num_cpus::get();

    let mut current_index = 0;
    let increment: usize =
        ((image_set_info.num_images as f32) / num_threads as f32).ceil() as usize;

    let image_size: usize = (image_set_info.image_width * image_set_info.image_height) as usize;

    let mut threads: Vec<thread::JoinHandle<Vec<Image>>> = Vec::with_capacity(num_threads);
    for _ in 0..num_threads {
        let next_index = std::cmp::min(
            image_set_info.num_images as usize,
            current_index + increment,
        );

        let image_path_string = String::from(image_path);
        let label_path_string = String::from(label_path);

        threads.push(thread::spawn(move || {
            let mut images: Vec<Image> = Vec::with_capacity(next_index - current_index);
            let image_file = read_file_part(
                &image_path_string,
                (16 + current_index * image_size) as u64,
                (next_index - current_index) * image_size,
            );
            let label_file = read_file_part(
                &label_path_string,
                (8 + current_index) as u64,
                next_index - current_index,
            );
            for index in 0..(next_index - current_index) {
                images.push(load_image(
                    &image_file,
                    &label_file,
                    index,
                    image_set_info.image_width as usize,
                    image_set_info.image_height as usize,
                ));
            }
            images
        }));

        current_index = next_index;
    }

    let mut image_set = Vec::with_capacity(image_set_info.num_images as usize);
    for th in threads {
        let images = th.join().expect("Error loading images");
        image_set.extend(images);
    }

    Ok(image_set)
}

fn load_image(
    image_bytes: &Vec<u8>,
    label_bytes: &Vec<u8>,
    index: usize,
    image_width: usize,
    image_height: usize,
) -> Image {
    let mut image = Image {
        label: label_bytes[index],
        pixels: vec![Default::default(); image_width * image_height],
    };

    for y in 0..image_width {
        for x in 0..image_height {
            let i = y * image_width + x;
            image.pixels[i] = (image_bytes[(i + (image_width * image_height) * index) as usize]
                as NetworkValueType)
                / (255 as NetworkValueType);
        }
    }

    image
}

fn load_image_set_info(bytes: &[u8]) -> ImageDataSetInfo {
    ImageDataSetInfo {
        magic_num: u8_arr_to_u32(&bytes[0..4]),
        num_images: u8_arr_to_u32(&bytes[4..8]),
        image_width: u8_arr_to_u32(&bytes[8..12]),
        image_height: u8_arr_to_u32(&bytes[12..16]),
    }
}

fn load_label_set_info(bytes: &[u8]) -> LabelDataSetInfo {
    LabelDataSetInfo {
        magic_num: u8_arr_to_u32(&bytes[0..4]),
        num_images: u8_arr_to_u32(&bytes[4..8]),
    }
}

fn u8_arr_to_u32(bytes: &[u8]) -> u32 {
    ((bytes[0] as u32) << 24)
        + ((bytes[1] as u32) << 16)
        + ((bytes[2] as u32) << 8)
        + (bytes[3] as u32)
}

fn read_file_part_into_buffer(filename: &str, offset: u64, buffer: &mut [u8]) {
    let f = std::fs::OpenOptions::new()
        .read(true)
        .open(&filename)
        .expect("no file found");
    let reader = std::io::BufReader::new(f);
    seek_read(reader, offset, buffer).expect("An error occurred while reading file");
}

fn read_file_part(filename: &str, offset: u64, length: usize) -> Vec<u8> {
    let f = std::fs::OpenOptions::new()
        .read(true)
        .open(&filename)
        .expect("no file found");
    let reader = std::io::BufReader::new(f);
    let mut buffer: Vec<u8> = vec![0; length];
    seek_read(reader, offset, &mut buffer).expect("An error occurred while reading file");
    buffer
}

fn seek_read(mut reader: impl Read + Seek, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    reader.seek(std::io::SeekFrom::Start(offset))?;
    reader.read_exact(buf)?;
    Ok(())
}
