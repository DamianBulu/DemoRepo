import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
import tkinter as tk

from networkx.conftest import needs_matplotlib
from webcolors import names


def draw_digit(filename="digit.png"):
    """ Funcție care deschide o fereastră pentru a desena o cifră,
            salvează imaginea ca fișier și returnează array-ul procesat (28x28). """
    size=(280,280)
    img=Image.new("L",size,255)
    draw=ImageDraw.Draw(img)

    def paint(event):
        x1,y1=(event.x-10),(event.y-10)
        x2,y2=(event.x+10),(event.y+10)

        draw.ellipse([x1,y1,x2,y2],fill=0)
        canvas.create_oval(x1,y1,x2,y2,fill="black",width=5)

    root=tk.Tk()
    root.title("Draw a digit and close the window")
    canvas=tk.Canvas(root,width=280,height=280,bg="white")
    canvas.pack()
    canvas.bind("<B1-Motion>",paint)

    def save_and_exit():
        img.save(filename)
        root.destroy()
    btn=tk.Button(root,text="Save",command=save_and_exit)
    btn.pack()
    btn.mainloop()
    #Image processing
    img=img.resize((28,28))
    img_array=np.array(img)
    img_array=255-img_array
    img_array=img_array/255.0 #Normalization(0-1)

    return img_array.reshape(1,784) #Transformation into an array for the model

# Încărcarea weighturilor și biasurilor din fișierul .npz
def load_model(filename="model_weights.npz"):
        data = np.load(filename)
        W1 = data['W1']
        b1 = data['b1']
        W2 = data['W2']
        b2 = data['b2']
        print(f"Model loaded from {filename}")
        return W1, b1, W2, b2

def predict(X, W1, b1, W2, b2):
    _, A2 = forward(X, W1, b1, W2, b2)
    prediction = np.argmax(A2, axis=1)
    return prediction[0]

if __name__ == "__main__":
    # Step 1: Draw the digit and save it as a processed array
    digit_array = draw_digit("digit.png")

    # Step 2: Load the model weights and biases
    W1, b1, W2, b2 = load_model("model_weights.npz")

    # Step 3: Use the model to predict the digit
    predicted_digit = predict(digit_array, W1, b1, W2, b2)

    # Step 4: Show the predicted digit
    print(f"Predicted Digit: {predicted_digit}")

    # Optionally, display the drawn image for visual feedback
    plt.imshow(digit_array.reshape(28, 28), cmap="gray")
    plt.title(f"Predicted Digit: {predicted_digit}")
    plt.show()