# Water Prediction by Transformer Model

## Dataset

The operational data of a certain sewage treatment plant for one year (see Data.csv) is provided in hourly time precision. The data includes rainfall intensity near the sewage treatment plant, inflow rate, COD concentration, and ammonia nitrogen concentration of the influent wastewater.

The whole dataset contains 8760 pieces of data. We fist sample them into 1000 sequense which contains the data and label.

And before the training, we normalize the data so that its size is between 0 and 1.

## Model

We use positional encoding and transformer model to predict.

### Positional encoding
$$
\begin{align*}
\text{PE}{(pos, 2i)} &= \sin\left(\frac{pos}{10000^{2i/d{\text{model}}}}\right) \\
\text{PE}{(pos, 2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d{\text{model}}}}\right)
\end{align*}
$$

$\text{PE}_{(pos, 2i)}$ represents the element in the position encoding matrix at position $(pos, 2i)$.

$\text{PE}_{(pos, 2i+1)}$ represents the element in the position encoding matrix at position $(pos, 2i+1)$.

$pos$ represents the position in the input sequence, ranging from $0$ to $L-1$.

$i$ represents the dimension in the position encoding vector, ranging from $0$ to $d_{\text{model}}-1$.

$d_{\text{model}}$ represents the embedding dimension in the Transformer model.

$\text{PE}$ represents the position encoding matrix.

And the model is transformer.

<img src='/home/chenshuo/cs/water_pre/transformer.jpg'>

## Requirment

You can creat you own envirnment and run the code in the terminal:

```
pip install -r requirements.txt
```


## Start Your Training
You can run the following code directly for your training
```
bash run.sh
```

And you can change the hyperparameters in the `run.sh`.

After trainig, you will get the model file `./model.pth`. Then you can run the `predict.py` by 

```
python predict.py
```
And you will get the result figure `./output.png`. Also, the output data can be found in `./output.txt`.

## Use my pretain

I also provide my trained file `model.pth`, and you can directly run the `predict.py` can get the result.

## Result
<img src=/home/chenshuo/cs/water_pre/output.png>