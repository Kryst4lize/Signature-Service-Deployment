
import tensorflow as tf
import tf2onnx
import onnx
from onnxsim import simplify

def convert_and_simplify(keras_path: str, onnx_path: str, input_name: str, output_name: str):
    print(f"\n--- Converting {keras_path} ---")
    
    # 1. Load the Keras model
    model = tf.keras.models.load_model(keras_path)
    
    # 2. Feed Keras exactly what it expects (Channels-Last: NHWC) to bypass the ValueError
    spec = (tf.TensorSpec((1, 224, 224, 3), tf.float32, name=input_name),)
    
    # 3. Convert to ONNX. 
    # The 'inputs_as_nchw' flag tells the exporter to wrap the input with a transpose, 
    # exposing a (1, 3, 224, 224) interface to Triton.
    print("Exporting to ONNX...")
    model_proto, _ = tf2onnx.convert.from_keras(
        model, 
        input_signature=spec, 
        opset=13,
        inputs_as_nchw=[input_name]  # <-- The crucial fix
    )
    
    # 4. Simplify the graph (This helps resolve the Integer Overflow bug)
    print("Simplifying ONNX graph...")
    model_simp, check = simplify(model_proto)
    if not check:
        print("Warning: Simplified ONNX model could not be validated!")
    else:
        print("Model simplified successfully.")
        
    # 5. Save the final optimized model
    onnx.save(model_simp, onnx_path)
    print(f"Saved optimized model to: {onnx_path}")

if __name__ == "__main__":
    # Convert VGG16
    convert_and_simplify(
        keras_path="original_model/vgg16_extractor.keras", 
        onnx_path="simple/vgg16_extractor.onnx", 
        input_name="input_layer",
        output_name="fc1"
    )
    
    # Convert ResNet50
    convert_and_simplify(
        keras_path="original_model/resnet50_extractor.keras", 
        onnx_path="simple/resnet50_extractor.onnx", 
        input_name="input_layer_1",
        output_name="fc1"
    )