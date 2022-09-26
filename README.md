# Generate Vice Headlines with Bloom
[![Try it here](https://img.shields.io/badge/%F0%9F%A4%97-Try%20it%20here!-yellow)](https://huggingface.co/spaces/marcderbauer/vice-headlines)
  
## :runner: Quickstart
  <b> not implemented yet :disappointed_relieved:</b>  
  If you can't be bothered to read all of this, you can just run
  ```
  chmod +x run.sh
  ./run.sh
  ```

## :snowflake: Context
This project originally started out as an RNN, which I wanted to implement in Pytorch. 
I had difficulties getting the model to create a coherent output. As I lacked reference values for training, I decided to finetune an existing model -- BLOOM. I hoped to learn more about the text-generation process from a top-down perspective, and to gather reference values for training in a "best-case" scenario.    

## :robot: Setup
#### 1. Install the required dependencies  
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
> If you decided to use the data included in the repository, you can skip this section.  

#### 1. Collecting the data  
  Assuming you setup the YouTube API correctly, all you need to do is run the [youtube/query_api.py](youtube/query_api.py).  
  You need to supply the requested channel's playlistId as an argument when launching the program.  
  For VICE and VICE News the respective commands are:
  ```
  python3 youtube/query_api.py UUn8zNIfYAQNdrFRrr8oibKw
  python3 youtube/query_api.py PLw613M86o5o7q1cjb26MfCgdxJtshvRZ-
  ```
  In order to find a channel's playlistId you need to  
  <ol>
    <li>Go to the channel
    <li>Find a playlist with all the channel's videos included (often the first playlist)
    <li>Click <em>PLAY ALL</em>
    <li>Copy everything after <code>list=</code> from the link
  </ol>
  
#### 2. Cleaning the data  
  In order to clean the data, you just need to run the [preprocess.py](preprocess.py).
  Assuming the file to process is called <code>vice.txt</code>:
  ```
  python3 preprocess.py vice.txt
  ```
    
## :chart_with_downwards_trend: Training
- How to run this, maybe some hyperparams

## :moyai: Inference
- How to use inference. Also add argparse

## :recycle: Conclusion
- How did this help me with RNN
- What did I learn in General?
