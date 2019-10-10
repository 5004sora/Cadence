
# coding: utf-8

# In[1]:


import spacy
import numpy as np

nlp = spacy.load('en')

def select_tags(inp):
    forb_V = ["are","is","am","be"]
    tags = []
    doc = nlp(inp)
    for token in doc:
        if (token.pos_ == "NOUN" and (token.dep_ == "pobj" or token.dep_ == "dobj")):
            tags.append(token.text)
        elif (token.pos_ == "ADJ" and token.dep_ == "acomp"):
            tags.append(token.text)
        elif token.pos_ == "VERB" and (token.dep_ == "xcomp" or token.dep_ == "ROOT") and token.text not in forb_V:
            tags.append(token.text)

    for chunk in doc.noun_chunks:
        tags.append(chunk.text)

    for ent in doc.ents:
        tags.append(ent.text)

    return tags

def NLPAlg(interest,schemes):
    print("\nStarting algorithm...")
    print("\nUser input: "+interest+"\n")
    print("Test sponsor schemes: "+' | '.join(schemes)+"\n")
    print("Matching with relevant schemes...\n")
    sims = []
    tags1 = select_tags(interest)
    join_tags1 = ' '.join(tags1)
    doc = nlp(join_tags1)
    for scheme in schemes:
        tags = select_tags(scheme)
        join_tags = ' '.join(tags)
        doc2 = nlp(join_tags)
        sim = doc.similarity(doc2)
        sims.append(sim)
        final_list = [x for _, x in sorted(zip(sims,schemes),reverse=True)]
    if len(final_list) >= 3:
        for i in range(3):
            print(str(i+1)+". "+final_list[i])
    else:
        short_final_list = []
        for i in range(len(final_list)):
             short_final_list.append(final_list[i])
        for j in range(len(short_final_list)):
            print(str(j+1)+". "+short_final_list[j])

# In[2]:


#Test case 1:

NLPAlg("Interested in investing in a hedgefund.",["Invest in a hedgefund today! You'll enjoy it!","You are next."])


# In[3]:


#Test case 2:

NLPAlg("I am a finance student and would like to get started with an internship. I am a hard worker.",
["Looking for anyone who is interested in computing and machine learning technologies.",
 "Feel free to contact me if you are a hard worker and have a keen business sense.",
 "We have a great introductory programme to financial development",
 "We have a couple of vacancies in the toilet cleaning profession",
 "If you're an eager learner and don't have much experience with coding, we have the perfect opportunity for you!",
 "Busy this summer? Looking for some work experience before going off to uni? We can help with that!"
 "We are offering a software development intership this Summer",
 "This internship concerns anyone looking at applications of technology in different industries",
 "Anyone willing to have a steep learning curve over the Summer is welcome on this course ",
 "This test case should never actually be returned",
 "I am interested in sponsoring anyone with a strong idea"])


# In[4]:


#Test case 3:

NLPAlg("I am a literature student from China looking to rack up some good experience speaking English. I would love to teach it somewhere.",
["We are offering paid internships for young students. Some time along the way you could get a trip to China!",
"A law firm offering work experience this summer. We look forward to meeting you and we especially welcome anyone with a background in literature.",
"We are a technology company who are interested in keen learners and passionate coders. We look forward to hearing from you.",
"Originally based in Scotland, our company will take you to new heights with our new sports programme!"])


# In[5]:


#Test case 4:

NLPAlg("I am an up-and-coming Taiwanese tycoon, and I would like to receive funding for a new company that I'm thinking of starting. I think it'll help the economy of Taiwan and I hope you pick me.",
       ["We would like to offer you funding if you could prove your contribution to the Taiwanese economy.",
       "I am looking to fund a project in the South-East Asian economic landscape. I will consider you.",
       "This scheme will only work if you let us know that your company will work in healthcare.",
       "There is great potential in your future plans, and I would like to be a part of them. Although, I was in prison for five years."])
