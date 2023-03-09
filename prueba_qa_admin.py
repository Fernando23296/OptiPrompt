import os

val_trusted_list = ["False", "True"]
val_target_list = ["no", "yes"]
val_question_list = ["The Answer Is"]
i = 0
os.system(f"python main_qa.py --trusted {val_trusted_list[i]}  --target {val_target_list[i]} --question '{val_question_list[0]}'" )