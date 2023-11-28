import os
import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub("wine_daily")

   image = modal.Image.debian_slim().pip_install(["hopsworks", "numpy"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("id2223"))
   def f():
       g()


def generate_wine(quality, type_percent, vol_acid_mean, vol_acid_std, citric_acid_mean, citric_acid_std,
                    chlorides_mean, chlorides_std, sulfur_mean, sulfur_std, density_mean, density_std,
                  alcohol_mean, alcohol_std):
    """
    Returns a single iris flower as a single row in a DataFrame
    """
    import pandas as pd
    import numpy as np

    # We model each wine feature as being a gaussian with the mean and std as given by the data of each quality.
    # This is obviously a simplification, since a few of the distributions are multi-modal (gaussian) distributions,
    # but this suffices as an approximation and applies well to most variables.
    df = pd.DataFrame({"type": [int(np.random.choice(2, 1, p=[1-type_percent, type_percent])[0])],
                       "volatile_acidity": [vol_acid_mean + vol_acid_std * np.random.randn()],
                       "citric_acid": [citric_acid_mean + citric_acid_std * np.random.randn()],
                       "chlorides": [chlorides_mean + chlorides_std * np.random.randn()],
                       "total_sulfur_dioxide": [sulfur_mean + sulfur_mean * np.random.randn()],
                       "density": [density_mean + density_std * np.random.randn()],
                       "alcohol": [alcohol_mean + alcohol_std * np.random.randn()]
                      })
    df['quality'] = quality
    return df


def get_random_wine_quality():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    high_quality_df = generate_wine(2, 0.17, 0.29, 0.12, 0.33, 0.11, 0.045, 0.021, 110, 47, 0.9930, 0.0030, 11.4, 1.2)
    good_quality_df = generate_wine(1, 0.22, 0.31, 0.15, 0.32, 0.14, 0.054, 0.031, 115, 55, 0.9946, 0.0030, 10.6, 1.1)
    low_quality_df = generate_wine(0, 0.31, 0.40, 0.19, 0.30, 0.16, 0.065, 0.043, 119, 62, 0.9958, 0.0025, 9.9, 0.84)

    # randomly pick one of these 3 and write it to the featurestore
    pick_random = random.randint(0, 2)
    if pick_random == 2:
        wine_df = high_quality_df
        print("High Quality Wine added")
    elif pick_random == 1:
        wine_df = good_quality_df
        print("Good Quality Wine added")
    else:
        wine_df = low_quality_df
        print("Low Quality Wine added")

    return wine_df


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_df = get_random_wine_quality()
    wine_fg = fs.get_feature_group(name="wine", version=1)
    wine_fg.insert(wine_df)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("iris_daily")
        with stub.run():
            f()
