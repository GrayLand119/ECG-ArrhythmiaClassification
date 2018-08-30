from PIL import Image
import os


if __name__ == "__main__":
    cur_dir = os.getcwd() + "/"
    items = os.listdir(cur_dir)

    png_files = list(filter(lambda x: x.endswith("png"), items))
    for png_file in png_files:
        png_filepath = cur_dir + png_file
        with Image.open(png_filepath) as image:
            jpeg_img = Image.new('RGB', image.size)
            jpeg_img.paste(image, image)
            jpeg_img.save(cur_dir + os.path.basename(png_filepath).split('.')[-2] + ".JPEG", format='JPEG')
