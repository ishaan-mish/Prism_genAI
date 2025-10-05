1~create a folder that will house your input txt and output files, then create an input and output folder it in, store the book as a txt file in that input folder,after you have created the text file, right click and select copy as path.

the path will be something like this

C:\Users\golec\Downloads\final fantasy

your folder name will be something different if you wish, but if you keep the same name you can copy and paste the commands in step2 making it easier for you.

2~First open anaconda, then open an anaconda prompt 

Execute these command line commands: 

replace golec with whatever your laptops user name is

cd C:\Users\golec\Downloads\final fantasy

conda create --name booknlp python=3.10
conda activate booknlp

pip install booknlp
python -m spacy download en_core_web_sm

3~next create a booknlps folder in users>your_user_name

a folder like this will be created

 C:\Users\golec\booknlps

4~then create 3 subfolders, but do not close the previous terminal, keep it in the background, I assume you have git installed

1)coref_google
open this empty sub folder, right click and in the drop down menu, and then click "open in terminal"

run this command:

git clone https://huggingface.co/google/bert_uncased_L-12_H-768_A-12

(then close the terminal,let the install finish coomplelelty)

2)entities_google
open this empty sub folder, right click and in the drop down menu, and then click "open in terminal"

run this command:

git clone https://huggingface.co/google/bert_uncased_L-6_H-768_A-12

(then close the terminal, after the install happens)

3)speaker_google
open this empty sub folder, right click and in the drop down menu, and then click "open in terminal"
run this command:

git clone https://huggingface.co/google/bert_uncased_L-12_H-768_A-12

(then close the terminal,after your installation is over)

5~now go here "C:\Users\golec\anaconda3\Lib\site-packages\booknlp\english\gender_inference_model_1.py"

of course your user name would be different

"C:\Users\your user name\anaconda3\Lib\site-packages\booknlp\english\gender_inference_model_1.py"

open in any python editor

ctrl+f and find this function: def read_hyperparams

the code block surrounding it looks like this

def read_hyperparams(self, filename):
 self.hyperparameters={}
 with open(filename) as file:
  header=file.readline().rstrip()
  gender_mapping={}
  for idx, val in enumerate(header.split("\t")[2:]):
   if val in self.genderID:
    gender_mapping[self.genderID[val]]=idx+2
 
change this line 

with open(filename) as file:

to 

with open(filename, encoding='UTF8') as file:

save it

~go into your final fantasy folder or whatever you have selected your folder name as and save the runner.py file there

~open the runner file and change these lines to your equivalent

# === Your actual input / output settings ===
input_file = Path(r"C:\Users\golec\Downloads\final fantasy\INPUT\NOVEL.txt")
output_directory = Path(r"C:\Users\golec\Downloads\final fantasy\OUTPUT")
raw_book_id = "a study in scarlet (actuak 8npuys)"  


replace the "C:\User....." for both with your actual paths

~when you installed boonlp, a booknlp_models folder would have been created

with a path like this

"C:\Users\golec\booknlp_models"

and contain 3 .model files, copy each path and replace those paths in these lines

coref_path  = Path(r"C:\Users\golec\booknlp_models\coref_google_bert_uncased_L-12_H-768_A-12-v1.0.model")
entity_path = Path(r"C:\Users\golec\booknlp_models\entities_google_bert_uncased_L-6_H-768_A-12-v1.0.model")
quote_path  = Path(r"C:\Users\golec\booknlp_models\speaker_google_bert_uncased_L-12_H-768_A-12-v1.0.1.model")

replace these lines in the runner code(provided in GitHub) with the paths of the respective files and save the file

also replace the input_file = Path(r"C:\Users\golec\Downloads\final fantasy\INPUT\NOVEL.txt") with your actual novel path.

~run the python file in the terminal with this command

python runner.py




