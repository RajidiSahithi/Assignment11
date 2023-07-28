def model_summary(model,input_size):
    model = ResNet18().to(device)
    summary(model, input_size=(3, 32, 32))
    return model,input_size