import os
import modal
    
LOCAL=False

# Use this function to run on modal once every day
if LOCAL == False:
   stub = modal.Stub("wine_batch")
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","scikit-learn","dataframe-image", 'numpy', 'tensorflow'])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def g():
    import pandas as pd
    import hopsworks
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import keras
    import numpy as np

    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=1)
    model_dir = model.download()

    # Load the model and the weights

    n_input = 7  # Number of features
    n_hidden = 256  # Number of hidden nodes
    n_out = 3  # Number of classes
    model = keras.Sequential([
        keras.layers.Dense(n_hidden, input_dim=n_input, activation='selu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(n_hidden, activation='selu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(n_hidden, activation='selu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(n_hidden, activation='selu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(n_hidden, activation='selu'),
        keras.layers.Dense(n_out, activation='softmax')
    ])
    model.load_weights(model_dir + "/wine_model.h5")

    # Get the latest training data, in this case just the initial training data
    feature_view = fs.get_feature_view(name="wine", version=1)
    batch_data = feature_view.get_batch_data()

    # Do the model predictions
    y_pred = model.predict(batch_data)

    # Pick a random offset and choose that prediction to use in the history png
    wine_quality = y_pred[-1]
    dataset_api = project.get_dataset_api()
   
    wine_fg = fs.get_feature_group(name="wine", version=1)
    df = wine_fg.read()
    label = df.iloc[-1]["quality"]

    # Get the previous wine predictions
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Wine quaility Prediction/Outcome Monitoring"
                                                )

    # Add the new wine prediction to the history
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [wine_quality],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})

    # Create an image of the history to display on the app
    history_df = monitor_fg.read()
    history_df = pd.concat([history_df, monitor_df], ignore_index=True)

    # Create a clear prediction history
    history_df['label'] = history_df.label.apply(lambda q: 'Low Quality' if q == 0
                                        else 'Good Quality' if q == 1 else 'High Quality')
    history_df['prediction'] = history_df.prediction.apply(lambda q: np.argmax(q))
    history_df['prediction'] = history_df.prediction.apply(lambda q: 'Low Quality' if q == 0
                                        else 'Good Quality' if q == 1 else 'High Quality')

    df_recent = history_df.tail(8)
    dfi.export(df_recent, './df_recent.png', table_conversion='matplotlib')
    dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)

    # Create a confusion matrix of the last 500 elements in the training data.
    predictions = history_df.prediction.apply(lambda q: 0 if q == 'Low Quality'
                                                        else 1 if q == 'Good Quality' else 2)
    labels = history_df.label.apply(lambda q: 0 if q == 'Low Quality'
                                                        else 1 if q == 'Good Quality' else 2)

    # Take the 500 last elements in the training set and produce a confusion matrix, upload it to hopsworks.
    results = confusion_matrix(labels, predictions)

    df_cm = pd.DataFrame(results, ['True Low Quality', 'True Good Quality', 'True High Quality'],
                         ['Pred Low Quality', 'Pred Good Quality', 'Pred High Quality'])

    cm = sns.heatmap(df_cm, annot=True, fmt='g')
    fig = cm.get_figure()
    fig.savefig("./confusion_matrix.png")
    dataset_api.upload("./confusion_matrix.png", "Resources/images", overwrite=True)


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

