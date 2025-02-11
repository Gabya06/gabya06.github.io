from typing import List

from bs4 import BeautifulSoup
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk import RegexTokenizer
import re
from requests import HTTPError, Request
import TextBlob
from urllib.request import urlopen
import WordCloud

def get_page(url, headers) -> BeautifulSoup:
    try:
        req = Request(url, headers=headers)
        page = urlopen(req)
        soup = BeautifulSoup(page, "html.parser")
        return soup
    except HTTPError as e:
        print(f"Error opening page {e}")


def get_jobs() -> pd.DataFrame:
    # load webpage using Beautifulsoup first
    soup = get_page()
    table = soup.find("table")
    table_rows = table.find_all("tr")

    data_dict = {}
    data_list = []

    # loop through table rows and extract information to dictionary
    for elem in table_rows[1:]:
        try:
            data_dict["company"] = elem.find("td", {"class": "company"}).text
        except AttributeError:
            data_dict["company"] = None
        try:
            data_dict["job_title"] = elem.find("td", {"class": "job_title"}).text
        except AttributeError:
            data_dict["job_title"] = None
        try:
            data_dict["job_url"] = elem.find("a", {"class": "jobLink"})["href"]
        except AttributeError:
            data_dict["job_url"] = None

        data_list.append(data_dict)
        data_dict = {}
    # put results in dataframe
    data = pd.DataFrame(data_list)
    # add column with job listing Id number
    data = data.assign(
        job_id=data.job_url.map(
            lambda x: re.findall(pattern="jobListingId=[0-9]+", string=x)[0]
        ).map(lambda x: x.strip("jobListingId="))
    )
    # add column with url
    listing_base_url = f"https://www.glassdoor.com/job-listing/?jl="
    data = data.assign(
        url=data.job_id.map(lambda x: listing_base_url + str(x))
    )
    
    return data


def plot_wordcloud(word_counts: List):
    wordcloud = WordCloud(width = 300,
                        height = 300,
                        background_color='white',
                        max_font_size=50, max_words=150)

    wordcloud = wordcloud.generate_from_frequencies(word_counts)

    # plot words
    plt.figure(figsize=(6,4),facecolor = 'white', edgecolor='blue')
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title("Top Glassdoor.com words")
    plt.savefig("review_wordcloud.png")
    plt.show()


def print_polarity_subjectivity(df):
    sample_string = df.clean_review.iloc[0]
    polarity = TextBlob(sample_string).sentiment.polarity
    subjectivity = TextBlob(sample_string).subjectivity
    # polarity & subjectivity:
    print(f"Sample Review:\n{sample_string}\nTextBlob polarity:{polarity}"
        f" and subjectivity:{subjectivity}")


def tokenize_overview(mydata, overview_col):
    """
    Function to clean show overview

    """
    # removes punctuation
    tokenizer = RegexpTokenizer(r"\w+")
    stop_words = stopwords.words("English")

    # split text
    tokens = mydata[overview_col].map(lambda x: tokenizer.tokenize(x))
    # strip white spaces & lower case
    tokens = tokens.map(lambda x: [i.lower().strip("_") for i in x])
    # remove stop words
    tokens = tokens.map(lambda x: [i for i in x if i not in stop_words])
    # remove empty strings
    tokens = tokens.map(lambda x: [i for i in x if i != ''])

    return tokens