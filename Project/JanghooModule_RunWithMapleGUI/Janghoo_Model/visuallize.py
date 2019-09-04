import matplotlib.pyplot as plt
def plt_show_loss(history) :
    plt.figure(figsize = [10, 7])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


def plt_show_acc(history) :
    plt.figure(figsize = [10, 7])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc = 0)
    plt.show()
