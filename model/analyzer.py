import tensorflow as tf
import argparse



def main():
    parser = argparse.ArgumentParser(description='Analyze tflite model')
    parser.add_argument('-p', '--path', type=str, required=True)

    args = parser.parse_args()

    tf.lite.experimental.Analyzer.analyze(model_path=args.path)

if __name__ == "__main__" :
    main()