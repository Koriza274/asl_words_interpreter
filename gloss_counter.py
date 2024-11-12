import pandas as pd
import os

video_folder = "data/ASL_Citizen/ASL_Citizen/videos"
output_folder = "data/output"
output_train_summary = os.path.join(output_folder, "train_gloss_summary.csv")
output_val_summary = os.path.join(output_folder, "val_gloss_summary.csv")
output_test_summary = os.path.join(output_folder, "test_gloss_summary.csv")

train_df = pd.read_csv("data/ASL_Citizen/ASL_Citizen/splits/train.csv")
val_df = pd.read_csv("data/ASL_Citizen/ASL_Citizen/splits/val.csv")
test_df = pd.read_csv("data/ASL_Citizen/ASL_Citizen/splits/test.csv")

def count_videos_per_gloss(dataframe, output_csv):
    gloss_video_count = dataframe.groupby("Gloss").size().reset_index(name="Video Count")
    
    gloss_video_count.to_csv(output_csv, index=False)
    print(f"Gloss-Zählung gespeichert in {output_csv}")

print("Trainingsdaten:")
count_videos_per_gloss(train_df, output_train_summary)

print("\nValidierungsdaten:")
count_videos_per_gloss(val_df, output_val_summary)

print("\nTestdaten:")
count_videos_per_gloss(test_df, output_test_summary)
