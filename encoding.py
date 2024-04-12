import cv2
import numpy as np
from scipy.fftpack import dct, idct
import gradio as gr
# RLE Encoding
def rle_encode(data):
    encoding = []
    prev_char = data[0]
    count = 1
    for char in data[1:]:
        if char == prev_char:
            count += 1
        else:
            encoding.append((prev_char, count))
            count = 1
            prev_char = char
    encoding.append((prev_char, count))
    return encoding

# RLE Decoding
def rle_decode(data):
    decoding = []
    for char, count in data:
        decoding.extend([char] * count)
    return np.array(decoding, dtype=np.uint8)

# LZW Encoding
def lzw_encode(input_string):
    dictionary = {chr(i): i for i in range(256)}
    word = ""
    result = []
    dict_size = 256

    for char in input_string:
        symbol = chr(char)  # Convert to string for dictionary lookup
        wc = word + symbol
        if wc in dictionary:
            word = wc
        else:
            result.append(dictionary[word])
            dictionary[wc] = dict_size
            dict_size += 1
            word = symbol
    if word:
        result.append(dictionary[word])
    return result

# LZW Decoding
def lzw_decode(encoded_data):
    dictionary = {i: bytes([i]) for i in range(256)}  # Use bytes for dictionary values
    result = bytearray()
    prev_code = encoded_data.pop(0)
    result += dictionary[prev_code]
    word = dictionary[prev_code]
    for code in encoded_data:
        if code in dictionary:
            entry = dictionary[code]
        else:
            entry = word + word[:1]
        result += entry
        dictionary[len(dictionary)] = word + entry[:1]
        word = entry
    return bytes(result)

def display_encoded_as_image(encoded_data, original_shape):
    # Flatten the list if it's a list of tuples from RLE, for example
    if isinstance(encoded_data[0], tuple):
        flat_encoded = [item for sublist in encoded_data for item in sublist]
    else:
        flat_encoded = encoded_data

    # Scale the encoded data to fit in the 0-255 range
    max_val = np.max(flat_encoded)
    scaled_encoded = np.array(flat_encoded) * (255.0 / max_val)

    # Ensure the data is of type uint8
    scaled_encoded_uint8 = np.array(scaled_encoded, dtype=np.uint8)

    # Calculate a new shape for visualization if necessary
    new_shape_length = np.product(original_shape[:2])
    
    if len(scaled_encoded_uint8.flatten()) < new_shape_length:
        # If the encoded data is too short, pad it
        padded_encoded = np.pad(scaled_encoded_uint8, (0, new_shape_length - len(scaled_encoded_uint8)), 'constant')
    else:
        # Use as much of the encoded data as fits into the original shape
        padded_encoded = scaled_encoded_uint8[:new_shape_length]

    # Reshape for display
    display_image = np.reshape(padded_encoded, original_shape[:2])

    return display_image


def process_image_gradio(image, method):
    # Convert the PIL image to an OpenCV image
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    
    # Apply the chosen method
    if method == 'rle':
        encoded = rle_encode(image.flatten())
        decoded = rle_decode(encoded)
        encoded_image = display_encoded_as_image(encoded, image.shape)
        processed_image = decoded.reshape(image.shape)
    elif method == 'dct':
        encoded = dct(dct(image.T, norm='ortho').T, norm='ortho')
        
        encoded_image =  display_encoded_as_image(encoded, image.shape)
        processed_image = idct(idct(encoded.T, norm='ortho').T, norm='ortho')
    elif method == 'lzw':
        encoded = lzw_encode(image.tobytes())
        decoded = lzw_decode(encoded)
        encoded_image = display_encoded_as_image(encoded, image.shape)
        processed_image = np.frombuffer(decoded, dtype=np.uint8).reshape(image.shape)
    else:
        return None, "Unsupported method", None
    
    # Prepare the encoded data visualization
    
    
    # Convert processed and encoded images to compatible format for Gradio output
    processed_image = cv2.cvtColor(np.clip(processed_image, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    encoded_image = cv2.cvtColor(encoded_image, cv2.COLOR_GRAY2RGB)
    
    # Return the original, encoded visualization, and processed images
    return image, encoded_image, processed_image

# Create a Gradio interface
iface = gr.Interface(
    fn=process_image_gradio,
    inputs=[gr.inputs.Image(shape=(200, 200)), gr.inputs.Radio(['rle', 'dct', 'lzw'])],
    outputs=[gr.outputs.Image(type="numpy", label="Original Image"),
             gr.outputs.Image(type="numpy", label="Encoded Image Visualization"),
             gr.outputs.Image(type="numpy", label="Processed Image")],
    
    title="Image Encoding and Decoding",
    description="Upload an image and select an encoding method to see the original image, "
                "the visual representation of the encoded data, and the processed (decoded) image."
)

iface.launch()