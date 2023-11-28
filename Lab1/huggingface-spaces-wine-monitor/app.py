import gradio as gr
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()
wine_pred_fg = fs.get_feature_group(name="wine_predictions", version=1)
df = wine_pred_fg.read()

latest_pred = df['prediction'].iloc[-1]
latest_label = df['label'].iloc[-1]
latest_pred = {'High Quality': float(latest_pred[2]),
                'Good Quality': float(latest_pred[1]),
                'Low Quality': float(latest_pred[0])}
latest_label = 'Low Quality' if latest_label == 0 else 'Good Quality' if latest_label == 1 else 'High Quality'

dataset_api.download("Resources/images/df_recent.png", overwrite=True)
dataset_api.download("Resources/images/confusion_matrix.png", overwrite=True)

with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column():
          gr.Label("Today's Predicted Wine Quality")
          gr.Label(latest_pred, num_top_classes=3, )
      with gr.Column():          
          gr.Label("Today's Actual Wine Quality ")
          gr.Label(latest_label)
    with gr.Row():
      with gr.Column():
          gr.Label("Recent Prediction History")
          input_img = gr.Image("df_recent.png", elem_id="recent-predictions")
      with gr.Column():          
          gr.Label("Confusion Maxtrix with Historical Prediction Performance")
          input_img = gr.Image("confusion_matrix.png", elem_id="confusion-matrix")        

demo.launch()
