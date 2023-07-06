import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print("Loading datasets...  ", end = "")
gdf = pd.read_csv("Kaggle Files/12k_ld_imputed.csv")
tdf = pd.read_csv("Kaggle Files/quantitative_traits.csv")
gdf = gdf.rename(columns = {gdf.columns[0]: "ID"}).set_index("ID")
tdf = tdf.rename(columns = {tdf.columns[0]: "ID"}).set_index("ID").fillna(method = "ffill")
gdf = gdf[gdf.index.isin(tdf.index.values)]
print("Done")

enc_map = {0.0: "00", -1.0: "01", 1.0: "10", 2.0: "11"}
loc_map = list(gdf.columns)
enc_op = np.array(list(enc_map.keys()))

flatten = lambda arr: [val for subarr in arr for val in subarr]
take = lambda arr, i: [val[i] for val in arr]
decode = lambda arr: [decode_trait("".join(str(int(round(value))) for value in subarr)) for subarr in arr]
unpack = lambda arr: [list(val.values()) for val in arr]

def crand(val):
    val = float(random.randrange(0, 2) - 1)
    return enc_op[np.abs(enc_op - val).argmin()]

print("Cleaning up...  ", end = "")
cround = np.vectorize(lambda val: enc_op[np.abs(enc_op - val).argmin()] if val not in enc_op else val)
gdf = pd.DataFrame(cround(gdf.values), columns = gdf.columns, index = gdf.index)
print("Done")

trait_map = {}

print("COLUMN NAME\tMIN\tMAX\tRANGE\tMEAN\n---------------------------------------------")
for column in tdf.columns:
    print(f"{column}     \t{tdf[column].min()}\t{tdf[column].max()}\t{round(tdf[column].max() - tdf[column].min())}\t{round(tdf[column].mean())}")
    
    max_val = math.ceil(tdf[column].max()) - math.floor(tdf[column].min())
    #req_dec = not all(np.mod(x, 1) == 0 for x in tdf[column])
    req_dec = max_val < 15
    if (req_dec): max_val *= 10
    bit_length = round(math.log([n for n in [2**n for n in range(1, 100)] if n > max_val][0], 2))
    trait_map.update({column: {"bits": bit_length, "dec": req_dec}})

def encode_gene(row):
    string = ""
    for value in row:
        string += enc_map[float(value)]
    return string

def decode_gene(b_string):
    gene_map = {}
    for i in range(0, len(string), 2):
        target_encoded = string[i:i + 2]
        gene = list(enc_map.keys())[list(enc_map.values()).index(target_encoded)]
        gene_map.update({loc_map[round(i/2)]: gene})
    return gene_map

def encode_trait(row):
    string = ""
    for column in trait_map.keys():
        bits = trait_map[column]["bits"]
        dec = trait_map[column]["dec"]
        val = round((row[column] - math.floor(tdf[column].min()))*(10 if dec else 1))
        string += "{0:b}".format(val).zfill(bits)
    return string

def decode_trait(b_string):
    dec_map = {}
    p = 0
    for column in trait_map.keys():
        bits = trait_map[column]["bits"]
        dec = trait_map[column]["dec"]
        b_seq = b_string[p:p + bits]
        trait = int(b_seq, 2)/(10 if dec else 1) + math.floor(tdf[column].min())
        dec_map.update({column: trait})
        p += bits
    return dec_map

def plot_history(epochs, history, accuracy_label):
    plt.title("Training Accuracy and Loss")
    plt.plot(range(epochs), history.history[accuracy_label], label = "Training Accuracy")
    plt.plot(range(epochs), history.history["loss"], label = "Training Loss")
    plt.legend()
    plt.show()

    plt.title("Validation Accuracy and Loss")
    plt.plot(range(epochs), history.history["val_" + accuracy_label], label = "Validation Accuracy")
    plt.plot(range(epochs), history.history["val_loss"], label = "Validation Loss")
    plt.legend()
    plt.show()

def record_accuracy(predicted_labels, actual_labels):
    corrs = []
    for c in range(len(tdf.columns)):
        x_values = take(predicted_labels, c)
        y_values = take(actual_labels, c)
        corr = pd.Series(x_values).corr(pd.Series(y_values))
        corrs.append(corr)
    print("Pearson scores: ", corrs)

master = pd.DataFrame(columns = ["GENE_STRING", "TRAIT_STRING"])

print("Generating master...  ")
for index, row in tqdm(gdf.iterrows(), total = gdf.shape[0]):
    master.loc[index] = [encode_gene(row), encode_trait(tdf.loc[index])]
print("Done")

print(master.head())

print("Generating samples...")
b_inputs = master["GENE_STRING"].tolist()
b_outputs = master["TRAIT_STRING"].tolist()
print(f"{len(b_inputs)} input samples, {len(b_outputs)} output samples")


from sklearn.model_selection import train_test_split

print("Preparing model inputs and outputs... ", end = "")
inputs = np.array([[int(digit) for digit in string] for string in tqdm(b_inputs)])
outputs = np.array([[int(digit) for digit in string] for string in tqdm(b_outputs)])
train_inputs, test_inputs, train_outputs, test_outputs = train_test_split(inputs, outputs, test_size = 0.3, random_state = 42)

mdf = gdf.merge(tdf, on = "ID")
column_count = len(tdf.columns)
features = mdf.iloc[:, :-column_count].values
labels = mdf.iloc[:, -column_count:].values

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.3, random_state = 42)
padded_train_labels = np.array([np.concatenate((np.zeros(train_features.shape[1] - train_labels.shape[1]), val)) for val in tqdm(train_labels)])
padded_test_labels = np.array([np.concatenate((np.zeros(test_features.shape[1] - test_labels.shape[1]), val)) for val in tqdm(test_labels)])
print("Done")

from tensorflow import keras
from keras.callbacks import History
history = History()
epochs = 20

corrs = []
for trait in range(len(tdf.columns)):
    tr_labels = np.array(take(train_labels, trait)).squeeze()
    te_labels = np.array(take(test_labels, trait)).squeeze()

    model = keras.Sequential([
        keras.layers.Dense(128, activation = "relu", input_shape = (train_inputs.shape[1],)),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation = "relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer = "adam", loss = "mse", metrics = ["accuracy"])

    model.fit(train_inputs, tr_labels, epochs = epochs, batch_size = 16, validation_data = (test_inputs, te_labels), callbacks = [history])

    loss, accuracy = model.evaluate(test_inputs, te_labels)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)

    predictions = model.predict(test_inputs)
    predicted_labels = predictions.squeeze()
    actual_labels = te_labels.squeeze()
    print("Predicted Labels:", predicted_labels[0])
    print("Actual Labels:", actual_labels[0])

    plt.title(list(tdf.columns)[trait])
    plt.scatter(predicted_labels, actual_labels)
    plt.show()
    corr = pd.Series(predicted_labels).corr(pd.Series(actual_labels))
    print("Pearson score: ", corr)
    corrs.append(corr)

print("Pearson scores: ", corrs)