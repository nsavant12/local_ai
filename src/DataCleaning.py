import re

import pandas as pd

from pathlib import Path

DATE_PATTERN = r'[A-Za-z]{3} \d{1,2}, \d{4}\s+\d{1,2}:\d{2}:\d{2}\s?[AP]M'


def read_imssg(file_path: str) -> pd.DataFrame:
    # Define filtering patterns
    tapback_header = "Tapbacks:"
    game_pigeon = "GamePigeon message:"
    reaction_pattern = r'^\s*(Loved|Liked|Disliked|Laughed at|Emphasized|Questioned) by .+$'
    reply_context_message = "This message responded to an earlier message."
    attachment_missing = "Attachment missing!"
    system_event_pattern = rf'^\s*{DATE_PATTERN}.*\b(named the conversation|added .+ to the conversation\.|removed .+ from the conversation\.|left the conversation\.)'
    attachment_pattern = r'(/Attachments/|/StickerCache/|\.(heic|jpe?g|png|mov|mp4|pdf|caf|gif)\s*$)'
    email_pattern = r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    read_receipt_pattern = r'\s*\(Read by you after [^)]+\)'
    edited_pattern = r'^Edited .*? later: '

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Apply filters to remove unwanted lines
    filtered_lines = []
    skipping_preview = False
    for line in lines:
        if line == "\n":
            skipping_preview = False
        if skipping_preview:
            continue
        if re.search(url_pattern,line) or game_pigeon in line:
            skipping_preview = True
            continue
        if (
            attachment_missing not in line and
            game_pigeon not in line and
            tapback_header not in line and
            reply_context_message not in line and
            not re.search(reaction_pattern, line, re.IGNORECASE) and
            not re.search(system_event_pattern, line) and
            not re.search(attachment_pattern, line, re.IGNORECASE) and
            not re.search(email_pattern, line)
        ):
            line = re.sub(read_receipt_pattern, "", line)
            line = re.sub(edited_pattern, "", line)
            filtered_lines.append(line.strip())

    # Normalize content
    content = '\n'.join(filtered_lines)
    # Replace narrow no-break space (iOS specific)
    content = content.replace(' ', ' ')
    # Remove LRM and RLM characters (Left-to-Right Mark and Right-to-Left Mark)
    content = content.replace('‎', '').replace('‏', '')

    pattern = rf'({DATE_PATTERN})\n([^\n]+)\n(.*?)(?=\n{DATE_PATTERN}|\Z)'
    messages = re.findall(pattern, content, re.DOTALL)
    df = pd.DataFrame(messages, columns=['timestamp', 'sender', 'message'])
    df['message'] = df['message'].str.strip()

    timestamps = []
    for timestamp in df['timestamp']:
        try:
            timestamp = pd.to_datetime(timestamp, format='mixed', errors='coerce')
        except Exception as e:
            print(f"Error parsing timestamp '{timestamp}': {e}")
            timestamp = pd.NaT
        timestamps.append(timestamp)

    df['timestamp'] = timestamps
    df = df[df['message'] != ""]
    return df


def main() -> None:
    all_chats = {}
    data_directory = Path(__file__).resolve().parent.parent / "data" / "private" / "imessage"
    for file in data_directory.glob('*.txt'):
        file_name = file.stem
        df = read_imssg(file)
        #Filter out spam & promotional messages
        if (df['sender'] == "Me").any():
            all_chats[file_name] = df
    combined_text = ""
    for file_name, df in all_chats.items():
        combined_text += "\n".join(df['message']) + "\n"
    combined_text.count("Let's play")
    combined_text.count("Your move.")
    with open(Path(__file__).resolve().parent.parent / "output" / "combined_text.txt", "w") as f:
        f.write(combined_text)


if __name__ == "__main__":
    main()
