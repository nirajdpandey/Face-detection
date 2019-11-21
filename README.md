# Tensorflow-Face-detection
Detecting faces from olivetti_faces data set by using Tensor-flow

## Content
```
1. Visualizing the olivetti_faces data
2. Deviding data into train & test with 90-10% ratio
3. Changinf labels to categorical variable
4. training and test loss.
```
### Visualization of the data 
```python
def plot_face(ax, img, image_shape):
    vmax = max(img.max(), -img.min())
    ax.imshow(img.reshape(image_shape), cmap=plt.cm.gray,
              interpolation='nearest',
              vmin=-vmax, vmax=vmax)
    return ax
```
![faces](https://github.com/nirajdevpandey/Tensorflow-Face-detection/blob/master/Data/olivetti1.jpg)

Following are activation finctions which are widely in use.

![activation_function](https://github.com/nirajdevpandey/Tensorflow-Face-detection/blob/master/Data/activation_functions.png)

### Results

```python
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.plot(accuracy_train,'k-', label='Training Data',color ='r')
plt.plot(accuracy_test ,'k--', label='Test Data',color='g')
plt.xlabel("Epochs (in 10s)")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('accuracy.jpg')
plt.show()
```


![title](https://github.com/nirajdevpandey/Tensorflow-Face-detection/blob/master/Data/accuracy.jpg)
Hence, We have recieved 97% of accuracy on test set
