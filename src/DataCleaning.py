import json
import re

import pandas as pd

from pathlib import Path

if __package__:
    from .ConversationStructure import (
        build_reply_examples,
        split_sessions,
        split_train_val_sessions,
    )
else:
    from ConversationStructure import (
        build_reply_examples,
        split_sessions,
        split_train_val_sessions,
    )

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

def prep_examples(
    chats: dict[str, pd.DataFrame],
    self_sender: str = "Me",
    val_fraction: float = 0.1,
    seed: int = 42
) -> tuple[list[dict], list[dict]]:

    sessions = []
    for chat in sorted(chats):
        df = chats[chat]
        records = df.sort_values(
            "timestamp", kind="stable"
        ).to_dict("records")
        sessions.extend(split_sessions(records))

    good_sessions = [session for session in sessions if build_reply_examples(session, self_sender=self_sender)]
    train_sessions, val_sessions = split_train_val_sessions(good_sessions, val_fraction, seed)

    train_examples = []
    for session in train_sessions:
        train_examples.extend(
            build_reply_examples(session, self_sender=self_sender)
        )
    
    val_examples = []
    for session in val_sessions:
        val_examples.extend(
            build_reply_examples(session, self_sender=self_sender)
        )
    return train_examples, val_examples

def write_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")



def main() -> None:
    all_chats = {}
    data_directory = Path(__file__).resolve().parent.parent / "data" / "private" / "imessage"
    for file in data_directory.glob('*.txt'):
        file_name = file.stem
        df = read_imssg(file)
        #Filter out spam & promotional messages
        if (df['sender'] == "Me").any():
            all_chats[file_name] = df

    train_examples, val_examples = prep_examples(all_chats)
    output_dir = (
        Path(__file__).resolve().parent.parent / "output" / "conversations"
    )
    write_jsonl(train_examples, output_dir / "train.jsonl")
    write_jsonl(val_examples, output_dir / "validate.jsonl")

    print(f"Training ex: {len(train_examples)}")
    print(f"Validation ex: {len(val_examples)}")


if __name__ == "__main__":
    main()
