import matplotlib.pyplot as plt

def show_model_plot(history):
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.title('Model Accuracy')
    plt.ylabel("Accuracy")
    plt.xlabel('Epoch')
    plt.legend(['Train','Validation'])

    plt.show()

    # show loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])

    plt.show()