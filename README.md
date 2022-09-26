# Generate Vice Headlines with Bloom
[![Try it here](https://img.shields.io/badge/%F0%9F%A4%97-Try%20it%20here!-yellow)](https://huggingface.co/spaces/marcderbauer/vice-headlines)
  
## :runner: Quickstart
  <b> not implemented yet :disappointed_relieved:</b>  
  If you can't be bothered to read all of this, you can just run
  ```
  chmod +x run.sh     # Make run.sh executable
  ./run.sh            # Run the program
  ```

## :snowflake: Context
This project originally started out as an RNN I wanted to implement in Pytorch. 
I had difficulties getting the model to create a coherent output. As I lacked reference values for training, I decided to finetune an existing model -- BLOOM. I hoped to learn more about the text-generation process from a top-down perspective, and to gather reference values for training in a "best-case" scenario.    
  
  
## :robot: Setup
#### 1. Install the Required Dependencies  
    pip install -r requirements.txt
#### 2. Setup YouTube API
  > :heavy_exclamation_mark: This step is only necessary if you want to source the data yourself:heavy_exclamation_mark:   
  The dataset used to train the model is included under [/data/](/data/). It was collected 23.09.2022.  
    
  The data for this project is gathered through the [YouTube Data API v3](https://developers.google.com/youtube/v3).
  Setting up this API can roughly be divided into the following steps:  
  <ol>  
    <li>Create a Google Developer Account
    <li>Create a new project
    <li>Enable the YouTube Data API v3
    <li>Create credentials
    <li>Make the credentials accessible to your environment 
  </ol>  
    
  For in-depth guidance, please refer to this excellent [HubSpot Article](https://blog.hubspot.com/website/how-to-get-youtube-api-key).   
  
  
## :bar_chart: Data
> :heavy_exclamation_mark:If you decided to use the data included in the repository, you can skip this section.:heavy_exclamation_mark:  

#### 1. Collecting the Data  
  Assuming you setup the YouTube API correctly, all you need to do is run the [youtube/query_api.py](youtube/query_api.py). 
  It requires the name of your [client_secrets_file](https://github.com/marcderbauer/bloom/blob/27b80f7fbe63f463ca9941cb23454d78e55fed4b/youtube/query_api.py#L24).
  You need to supply the requested channel's playlistId as an argument when launching the program. It is possible to supply multiple playlistIds at once by seperating them with a space.    
  
  In order to find a channel's playlistId you need to  
  <ol>
    <li>Go to the channel
    <li>Find a playlist with all the channel's videos included (often the first playlist)
    <li>Click <em>PLAY ALL</em>
    <li>Copy everything after <code>list=</code> from the link
  </ol>

  Thus, the command to download all the titles for VICE and VICE News is:
  ```
  python3 youtube/query_api.py UUn8zNIfYAQNdrFRrr8oibKw PLw613M86o5o7q1cjb26MfCgdxJtshvRZ-
  ```

  
#### 2. Cleaning the Data  
  To clean the data, you just need to run the [preprocess.py](preprocess.py).
  Assuming the file to process is called <code>vice.txt</code>, the command is:
  ```
  python3 preprocess.py vice.txt
  ```
  By default, this removes non-english sentences, duplicates and entries consisting of less than three words.
  The resulting file is automatically split into sets of 80% train and 20% test in <code>/data/</code>.
    
## :chart_with_downwards_trend: Training
  Training can easily be run by executing the <code>main.py</code>.  
  If you have [Weights & Biases](https://wandb.ai) set up, you can uncomment the following [line](https://github.com/marcderbauer/bloom/blob/27b80f7fbe63f463ca9941cb23454d78e55fed4b/main.py#L72) to track your training:
  ```
  report_to="wandb"
  ```

## :moyai: Inference
  Inference can be run by executing <code>inference.py</code> with the prompt as argument. Furthermore, you can pass certain inference parameters as arguments e.g.:
  ```
  python3 inference.py North Korea --temp 0.42 --top_k 32 --rp 1.3
  
  Output:
  temp=0.42; k=32, p=0.92, rep=1.3:
  ----------------------------------------------------------------------------------------------------
  North Korea's 'Most Humane' Hospital
  ```
  Huggingface made a great [tutorial](https://huggingface.co/blog/how-to-generate) on different generation strategies, where each inference parameter is explained in depth.

## :recycle: Conclusion
  This project has been very insightful in gaining an understanding of text-generation from a top-down perspective. While implementing this project as a PyTorch RNN, I mostly scrambled around without having much of an understanding of what I was doing.  
  By fine-tuning BLOOM, I learned how to fine-tune an existing model, how to source data, how to pre-process it correctly and how to host the resulting model on [Hugging Face Hub](https://huggingface.co/spaces/marcderbauer/vice-headlines) with [Gradio](https://gradio.app/).
