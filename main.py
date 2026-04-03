import get_match as gm
import get_data as gd
import get_parse_ui as pu
from langchain_deepseek import ChatDeepSeek
import pandas as pd
import get_response as gr
results = []
import config

# Use Langchain to connect to Deepseek and enter the API key.
chat_model = ChatDeepSeek(model="deepseek-chat", api_key=config.api_key)

# Load PAM and the corpus.
df, PAMs = gd.load_PAMS_corpus()

# Iterate through each piece of corpus.
for index, row in df.iterrows():

    # Get the content of the user's question.
    user_message = row['Content'].lower()
    label = row['Label']

    # Parse user input
    parsed_input = pu.parse_user_input(chat_model, user_message)

    # Perform PAMs matching
    best_process_id, best_process_name, max_confidence = gm.get_match(
        task=parsed_input["task"],
        device_name=parsed_input["device_name"],
        device_id=parsed_input["device_id"],
        output_data=parsed_input["output_data"],
        models=PAMs
    )

    response = gr.get_response(best_process_id, chat_model, user_message, parsed_input["constraints"])

    results.append({
        'Content': row['Content'],
        'Label': label,
        'best_process_id': best_process_id,
        'best_process_name': best_process_name,
        'max_confidence': max_confidence,
        'response': response
    })

# 将结果列表转换为DataFrame
results_df = pd.DataFrame(results)

# 保存到Excel文件（移除了encoding参数）
results_df.to_excel('Response.xlsx', index=False)

print("The result has been saved to 'Response.xlsx'")