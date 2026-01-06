

try:
    from tflite_runtime.interpreter import Interpreter
    print("TFLite Interpreter found: tflite_runtime.interpreter.Interpreter")
except ImportError:
    try:
        from tensorflow.lite.python.interpreter import Interpreter
        print("TFLite Interpreter found: tensorflow.lite.python.interpreter.Interpreter")
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
            print("TFLite Interpreter found: tensorflow.lite.Interpreter")
        except ImportError as e:
            print("ERROR: No TFLite Interpreter found in your Python environment.")
            print(e)
            print("\nTry: 'pip install tensorflow' or 'pip install tensorflow-macos' (for Apple Silicon)")
