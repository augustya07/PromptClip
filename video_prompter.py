import concurrent.futures
import json
from dotenv import load_dotenv
from videodb import connect
from llm_agent import LLM, LLMType

load_dotenv()


def get_connection():
    """
    Get connection and load the env.
    :return:
    """
    conn = connect()
    return conn


def get_video(id):
    """
    Get video object
    :param id:
    :return:
    """
    conn = get_connection()
    all_videos = conn.get_collection().get_videos()
    video = next(vid for vid in all_videos if vid.id == id)
    return video

def chunk_doc(docs, chunk_size):
    """
    chunk document to fit into context of your LLM
    :param docs:
    :param chunk_size:
    :return:
    """
    for i in range(0, len(docs), chunk_size):
        yield docs[i: i + chunk_size]  # Yield the current chunk





def send_msg_openai(chunk_prompt, llm=LLM()):
    response = llm.chat(message=chunk_prompt)
    output = json.loads(response["choices"][0]["message"]["content"])
    return output


def send_msg_claude(chunk_prompt, llm):
    response = llm.chat(message=chunk_prompt)
    # TODO : add claude reposnse parser
    return response


def text_prompter(transcript_text, prompt, llm=None):
    chunk_size = 10000
    # sentence tokenizer
    chunks = chunk_doc(transcript_text, chunk_size=chunk_size)
    # print(f"Length of the sentence chunk are {len(chunks)}")

    if llm is None:
        llm = LLM()

    # 400 sentence at a time
    if llm.type == LLMType.OPENAI:
        llm_caller_fn = send_msg_openai
    else:
        # claude for now
        llm_caller_fn = send_msg_claude

    matches = []
    prompts = []
    i = 0
    for chunk in chunks:
        chunk_prompt = """
        You are a video editor who uses AI. Given a user prompt and transcript of a video analyze the text to identify sentences in the transcript relevant to the user prompt for making clips. 
        - **Instructions**: 
          - Evaluate the sentences for relevance to the specified user prompt.
          - Make sure that sentences start and end properly and meaningfully complete the discussion or topic. Choose the one with the greatest relevance and longest.
          - We'll use the sentences to make video clips in future, so optimize for great viewing experience for people watching the clip of these.
          - If the matched sentences are not too far, merge them into one sentence.
          - Strictly make each result minimum 20 words long. If the match is smaller, adjust the boundries and add more context around the sentences.

        - **Output Format**: Return a JSON list named 'sentences' that containes the output sentences, make sure they are exact substrings.
        - **User Prompts**: User prompts may include requests like 'find funny moments' or 'find moments for social media'. Interpret these prompts by 
        identifying keywords or themes in the transcript that match the intent of the prompt.
        """

        # pass the data
        chunk_prompt += f"""
        Transcript: {chunk}
        User Prompt: {prompt}
        """

        # Add instructions to always return JSON at the end of processing.
        chunk_prompt += """
        Ensure the final output strictly adheres to the JSON format specified without including additional text or explanations. \
        If there is no match return empty list without additional text. Use the following structure for your response:
        {
          "sentences": [
            {},
            ...
          ]
        }
        """
        prompts.append(chunk_prompt)
        i += 1

    # make a parallel call to all chunks with prompts
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_index = {
            executor.submit(llm_caller_fn, prompt, llm): prompt for prompt in prompts
        }
        for future in concurrent.futures.as_completed(future_to_index):
            try:
                result = future.result()
                if 'sentences' in result:
                    matches.extend(result['sentences'])
                else:
                    print(f"Warning: 'sentences' key not found in result: {result}")
            except Exception as e:
                print(f"Chunk failed to work with LLM {str(e)}")
    
    return matches





def send_msg_claude(chunk_prompt, llm):
    response = llm.chat(message=chunk_prompt)
    # TODO : add claude reposnse parser
    return response

def scene_prompter(transcript_text, prompt, llm=None):
    chunk_size = 20
    # sentence tokenizer
    chunks = chunk_doc(transcript_text, chunk_size=chunk_size)
    # print(f"Length of the sentence chunk are {(chunks)[0]}")

    llm_caller_fn = send_msg_openai
    if llm is None:
        llm = LLM()

    # 400 sentence at a time
    if llm.type == LLMType.OPENAI:
        llm_caller_fn = send_msg_openai
    else:
        # claude for now
        llm_caller_fn = send_msg_claude

    matches = []
    prompts = []
    i = 0

    for chunk in chunks:
        descriptions = [scene['description'] for scene in chunk]
        chunk_prompt = """
        You are a video editor who uses AI. Given a user prompt and AI-generated scene descriptions of a video, analyze the descriptions to identify segments relevant to the user prompt for creating clips.

        - **Instructions**: 
            - Evaluate the scene descriptions for relevance to the specified user prompt.
            - Ensure that selected segments have clear start and end points, covering complete ideas or actions.
            - Choose segments with the highest relevance and most comprehensive content.
            - Optimize for engaging viewing experiences, considering visual appeal and narrative coherence.

            - User Prompts: Interpret prompts like 'find exciting moments' or 'identify key plot points' by matching keywords or themes in the scene descriptions to the intent of the prompt.
        """

        chunk_prompt += f"""
        Descriptions: {json.dumps(descriptions)}
        User Prompt: {prompt}
        """


        chunk_prompt += """
         **Output Format**: Return a JSON list named 'result' that containes the  fileds `sentence`, `start`, `end` Ensure the final output
        strictly adheres to the JSON format specified without including additional text or explanations. \
        Output start and end time same as description
        If there is no match return empty list without additional text. Use the following structure for your response:
        {"result":{"descriptions":<>, "start":<>, "end":<>}}
        """
        prompts.append(chunk_prompt)
        i += 1
    
    for prompt in prompts:
      try:
        res = llm_caller_fn(prompt)
        # print(res)
        matches.append(res)
      except Exception as e:
        print(f"Chunk failed to work with LLM {str(e)}")
    return matches


def create_multimodal_chunks_individual(transcript, scenes, chunk_size=5):
    """
    Create multimodal data chunks for each individual scene within specified chunk size intervals.

    :param transcript: List of transcript entries, each with 'start', 'end', and 'text' fields.
    :param scenes: List of scene data, each with 'start', 'end', and 'description' fields.
    :param chunk_size: Number of scenes to group into each chunk. Default is 5.
    :return: List of multimodal data chunks.
    """
    def filter_text_by_time(transcript, start_time, end_time):
        result = []

        for entry in transcript:
            if float(entry['end']) > start_time and float(entry['start']) < end_time:
                text = entry['text']
                if text != '-':
                    result.append(text)

        return ' '.join(result)

    chunks = []

    for i in range(0, len(scenes), chunk_size):
        chunk = []
        for scene in scenes[i:i+chunk_size]:
            spoken = filter_text_by_time(transcript, float(scene["start"]), float(scene["end"]))
            data = {
                "visual": scene["description"], 
                'spoken': spoken, 
                'start': scene["start"], 
                'end': scene["end"]
            }
            chunk.append(data)
        chunks.append(chunk)

    return chunks


def multimodal_prompter(transcript, scene_index,prompt, llm=None):

    chunks = create_multimodal_chunks_individual(transcript, scene_index)


    if llm is None:
        llm = LLM()

    if llm.type == LLMType.OPENAI:
        llm_caller_fn = send_msg_openai
    else:
        llm_caller_fn = send_msg_claude

    matches = []
    prompts = []
    i = 0
    for chunk in chunks:


        chunk_prompt = f"""
       You are given visual and spoken information of the video of each second, and a transcipt of what's being spoken along with timestamp.
        Your task is to evaluate the data for relevance to the specified user prompt.
        Corelate visual and spoken content to find the relevant video segment.
        provide the start and end timestamps by analyse the full chunk and give longest matching timestamps. You can merge the 1 second chunks and transcripts to make continuous response.

        Multimodal Data:
        video: {chunk}
        User Prompt: {prompt}

    
        """
        chunk_prompt += """
         **Output Format**: Return a JSON list named 'result' that containes the  fileds `sentence`, `start`, `end` Ensure the final output
        strictly adheres to the JSON format specified without including additional text or explanations. \
        If there is no match return empty list without additional text. Use the following structure for your response:
        {"result":{"visual":<>,"spoken": <>, "start":<>, "end":<>}}
        """
        prompts.append(chunk_prompt)
        i += 1


    for prompt in prompts:
      try:
        res = llm_caller_fn(prompt)
        # print(res)
        matches.append(res)
      except Exception as e:
        print(f"Chunk failed to work with LLM {str(e)}")
    return matches

def extract_descriptions(data):
    descriptions = []
    for item in data:
        if 'result' in item:
            for result in item['result']:
                if 'descriptions' in result:
                    descriptions.append(result['descriptions'])
                elif 'description' in result:
                    descriptions.append(result['description'])
    return descriptions

def extract_clip_sentences(matches):
    sentences = []
    
    for index, item in enumerate(matches):
        try:
            # Assuming each item is a list with one dictionary element
            if isinstance(item, list) and len(item) > 0:
                item_dict = item[0]
            else:
                item_dict = item

            # Extracting the sentence
            sentence = item_dict['result']['spoken']
            sentences.append(sentence)

            # Optional: Print information about the processed item
            print(f"Processed Item {index}")

        except (TypeError, KeyError, IndexError) as e:
            print(f"Error processing item {index}: {e}")
            print(f"Item content: {item}")
            print("\n" + "-"*50 + "\n")

    return sentences


def extract_timestamps(data):
    timeframes = []
    for item in data:
        data_2 = item.get("result")
        for time in data_2:
            start_time = time.get('start')
            end_time = time.get('end')
            if start_time and end_time:
                start_seconds = convert_to_seconds(start_time)
                end_seconds = convert_to_seconds(end_time)
                timeframes.append({
                'start_time': start_seconds,
                'end_time': end_seconds
            })
            print(f"Extracted timeframe: start={start_seconds}, end={end_seconds}")  
    return timeframes

def convert_to_seconds(time):
    if isinstance(time, str):
        parts = time.split(':')
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        else:
            return int(time)
    elif isinstance(time, int):
        return time
    else:
        raise ValueError("Unsupported time format")

def merge_timeframes(timeframes):
    if not timeframes:
        print("No timeframes to merge")  
        return []
    
    sorted_timeframes = sorted(timeframes, key=lambda x: x['start_time'])
    merged_timeframes = [sorted_timeframes[0]]

    for timeframe in sorted_timeframes[1:]:
        last = merged_timeframes[-1]
        if last['end_time'] >= timeframe['start_time']:
            last['end_time'] = max(last['end_time'], timeframe['end_time'])
            print(f"Merging: {last}")  
        else:
            merged_timeframes.append(timeframe)
            print(f"Appending: {timeframe}")  

    return merged_timeframes

def extract_timestamps_multimodal(matches):
    timeframes = []
    for index, item in enumerate(matches):
        try:
            # Handle case where item might be None or not a dictionary
            if not item or not isinstance(item, dict):
                print(f"Skipping item {index}: Invalid or empty data")
                continue

            result = item.get('result')
            if not result or not isinstance(result, dict):
                print(f"Skipping item {index}: No valid 'result' found")
                continue

            spoken = result.get('spoken')
            start_time = result.get('start')
            end_time = result.get('end')

            if start_time is not None and end_time is not None:
                try:
                    start_seconds = convert_to_seconds(start_time)
                    end_seconds = convert_to_seconds(end_time)
                    timeframes.append({
                        'start_time': start_seconds,
                        'end_time': end_seconds,
                        'spoken': spoken
                    })
                    print(f"Extracted timeframe {index}: start={start_seconds}, end={end_seconds}")
                except ValueError as ve:
                    print(f"Skipping item {index}: Invalid time format - {ve}")
            else:
                print(f"Skipping item {index}: Missing start or end time")

        except Exception as e:
            print(f"Error processing item {index}: {e}")

    return timeframes





# def construct_timeline_and_stream( selected_frames):
#     timeline = Timeline(conn)
    
#     print("Selected frames:", selected_frames)  

#     for frame in selected_frames:
#         print("Adding frame:", frame)  
#         video_asset = VideoAsset(
#             asset_id=video_id,
#             start=frame['start_time'],
#             end=frame['end_time']
#         )
#         timeline.add_inline(video_asset)
    
#     print("Timeline created:", timeline)  
    
#     stream_url = timeline.generate_stream()
#     print("Stream URL:", stream_url)
#     return play_stream(stream_url)


