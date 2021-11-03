import G6_iris_recognition

# setup training paths
train_database_path = "Input_database/"
train_encoding_model_path = "encodingModel/irisEncodings.pickle"

# train model
G6_iris_recognition.iris_model_train(train_database_path,train_encoding_model_path)

# setup testing paths
test_encoding_model_path = "encodingModel/irisEncodings.pickle"
test_image_path = "test_images/parker_test.jpg"

# perform inference
iris_name = G6_iris_recognition.iris_model_test(test_encoding_model_path, test_image_path) 
print("Predicted Iris Name:", iris_name)
print("Actual Iris Name: Parker")
