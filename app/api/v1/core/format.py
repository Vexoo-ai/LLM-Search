import re
import datetime
import pytz
import dateutil
import pandas as pd


def is_date(string, fuzzy=False):
  # Parse a string into a date and check its validity
  try:
      dateutil.parser.parse(string, fuzzy=fuzzy)
      return True
  except ValueError:
      return False

def get_current_date():
    current_date = datetime.datetime.now(
        pytz.timezone("Asia/Kolkata")
    ).strftime("%B %d, %Y")
    return current_date
    
def format_date(d):
  # Standardize the date format for each search result
  current_date = get_current_date()
  date = dateutil.parser.parse(current_date, fuzzy=True).strftime("%b %d, %Y")
  if d is None:
    return None

  for t in ["second", "minute", "hour"]:
    if f"{t} ago" in d or f"{t}s ago" in d:
      return date

  t = "day"
  if f"{t} ago" in d or f"{t}s ago" in d:
    # n_days = int(re.search("(\d+) days? ago", d).group(1))
    n_days = int(re.search(r"(\d+) days? ago", d).group(1))

    return (
        datetime.datetime.strptime(date, "%b %d, %Y")
        - datetime.timedelta(days=n_days)
    ).strftime("%b %d, %Y")

  try:
    return dateutil.parser.parse(d, fuzzy=True).strftime("%b %d, %Y")
  except ValueError:
    for x in d.split():
      if is_date(x):
        return dateutil.parser.parse(x, fuzzy=True).strftime("%b %d, %Y")


def extract_source_webpage(link):
  # Extract source webpage
  return (
      link.strip()
      .replace("https://www.", "")
      .replace("http://www.", "")
      .replace("https://", "")
      .replace("http://", "")
      .split("/")[0]
  )


def simplify_displayed_link(displayed_link):
  # Simplify displayed link
  if displayed_link is None:
    return None
  return extract_source_webpage(displayed_link.split(' › ')[0])


def format_search_results(search_data, title_field=None, highlight_field=None):
  # Standardize search results as shown in Figure 3 (left) in the paper
  field = 'snippet_highlighted_words'
  if field in search_data and isinstance(search_data[field], list):
    search_data[field] = ' | '.join(search_data[field])

  field = 'displayed_link'
  if field in search_data:
    search_data[field] = simplify_displayed_link(search_data[field])

  # edge case 1
  if search_data.get('type') == 'local_time':
    source = search_data.get('displayed_link')
    date = format_date(search_data.get('date'))
    title = search_data.get('title')

    snippet = search_data.get('snippet')
    if snippet is None and 'result' in search_data:
      if 'extensions' in search_data and isinstance(
          search_data['extensions'], list
      ):
        snippet = '\n\t'.join(
            [search_data['result']] + search_data['extensions']
        )
      else:
        snippet = search_data['result']

    highlight = search_data.get('snippet_highlighted_words')
    if highlight is None and 'result' in search_data:
      highlight = search_data['result']

  # edge case 2
  elif 'type' in search_data and search_data['type'] == 'population_result':
    source = search_data.get('displayed_link')
    if source is None and 'sources' in search_data:
      if (
          isinstance(search_data['sources'], list)
          and 'link' in search_data['sources'][0]
      ):
        source = extract_source_webpage(search_data['sources'][0]['link'])

    date = format_date(search_data.get('date'))
    if date is None and 'year' in search_data:
      date = format_date(search_data['year'])

    title = search_data.get('title')

    snippet = search_data.get('snippet')
    if snippet is None and 'population' in search_data:
      if 'place' in search_data:
        snippet = '\n\t'.join(
            [
                f"{search_data['place']} / Population",
            ]
            + [
                search_data['population'],
            ]
        )
      else:
        snippet = search_data['population']

    highlight = search_data.get('snippet_highlighted_words')
    if highlight is None and 'population' in search_data:
      highlight = search_data['population']

  else:
    source = search_data.get('displayed_link')
    date = format_date(search_data.get('date'))
    title = (
        search_data.get('title')
        if title_field is None
        else search_data.get(title_field)
    )
    highlight = (
        search_data.get('snippet_highlighted_words')
        if highlight_field is None
        else search_data.get(highlight_field)
    )
    snippet = search_data.get('snippet', '')

    if 'rich_snippet' in search_data:
      for key in ['top', 'bottom']:
        if (
            key in search_data['rich_snippet']
            and 'extensions' in search_data['rich_snippet'][key]
        ):
          snippet = '\n\t'.join(
              [snippet] + search_data['rich_snippet'][key]['extensions']
          )

    if 'list' in search_data:
      assert isinstance(search_data['list'], list)
      snippet = '\n\t'.join([snippet] + search_data['list'])

    if 'contents' in search_data and 'table' in search_data['contents']:
      tbl = search_data['contents']['table']
      assert isinstance(tbl, list)
      snippet += '\n'
      for row in tbl:
        snippet += f'\n{",".join(row)}'

    if snippet is not None and snippet.strip() == '':
      snippet = None

  return {
      'source': source,
      'date': date,
      'title': title,
      'snippet': snippet,
      'highlight': highlight,
  }


def format_knowledge_graph(search_data):
  # Standardize knowledge graphs as shown in Figure 3 (left) in the paper
  source = None
  if "source" in search_data and "link" in search_data["source"]:
    source = extract_source_webpage(search_data["source"]["link"])

  date = None

  title = None
  if "title" in search_data:
    title = search_data["title"]
    if "type" in search_data:
      title += f"\n\t{search_data['type']}"

  snippet = ""
  for field in search_data:
    if (
        (field not in ["title", "type", "kgmid"])
        and ("_link" not in field)
        and ("_stick" not in field)
        and isinstance(search_data[field], str)
        and not search_data[field].startswith("http")
    ):
      snippet += f"\n\t{field}: {search_data[field]}"

  if snippet.strip() == "":
    snippet = None
  else:
    snippet = snippet.strip()

  highlight = None

  return {
      "source": source,
      "date": date,
      "title": title,
      "snippet": snippet,
      "highlight": highlight,
  }


def format_questions_and_answers(search_data):
  # Standardize questions and answers as shown in Figure 3 (left) in the paper
  source = None
  if "link" in search_data:
    source = extract_source_webpage(search_data["link"])

  date = None

  title = None
  if "question" in search_data:
    title = search_data["question"]

  snippet = None
  if "answer" in search_data:
    snippet = search_data["answer"]

  highlight = None

  return {
      "source": source,
      "date": date,
      "title": title,
      "snippet": snippet,
      "highlight": highlight,
  }


def freshprompt_format(
    question,
    search_data,
    # ranked_results,
    reasoning_and_answer,
    num_organic_results,
    num_related_questions,
    num_questions_and_answers,
    num_retrieved_evidences,
):
  """Build FreshPrompt for each question

  Args:
    question: The question to process.
    search_data: Search data.
    reasoning_and_answer: The reasoning and answer.
    num_organic_results: Number of organic results to keep.
    num_related_questions: Number of related questions to keep.
    num_questions_and_answers: Number of questions and answers to keep.
    num_retrieved_evidences: Number of retrieved evidences to keep.

  Returns:
    A prompt that incorporates retrieved evidences for each question.
  """

  df = pd.DataFrame(columns=['source', 'date', 'title', 'snippet', 'highlight'])

  # Organic results
  organic_results = [None] * num_organic_results
  for k in range(num_organic_results):
    if (
        'organic_results' in search_data
        and len(search_data['organic_results']) > k
    ):
      organic_results[k] = format_search_results(
          search_data['organic_results'][k]
      )
    else:
      organic_results[k] = format_search_results({})

  for d in organic_results[::-1]:
    df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)

  # Related questions
  related_questions = [None] * num_related_questions
  for k in range(num_related_questions):
    if (
        'related_questions' in search_data
        and len(search_data['related_questions']) > k
    ):
      related_questions[k] = format_search_results(
          search_data['related_questions'][k], title_field='question'
      )
    else:
      related_questions[k] = format_search_results({})

  for d in related_questions[::-1]:
    df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)
    print(d['title'])

  # Questions and Answers
  questions_and_answers = [None] * num_questions_and_answers
  for k in range(num_questions_and_answers):
    if (
        'questions_and_answers' in search_data
        and len(search_data['questions_and_answers']) > k
    ):
      questions_and_answers[k] = format_questions_and_answers(
          search_data['questions_and_answers'][k]
      )
    else:
      questions_and_answers[k] = format_questions_and_answers({})

  for d in questions_and_answers[::-1]:
    df = pd.concat([df, pd.DataFrame([d])], ignore_index=True)

  # Knowledge graph
  knowledge_graph = None
  if 'knowledge_graph' in search_data:
    knowledge_graph = format_knowledge_graph(search_data['knowledge_graph'])
  else:
    knowledge_graph = format_knowledge_graph({})
  df = pd.concat([df, pd.DataFrame([knowledge_graph])], ignore_index=True)

  # Answer box
  answer_box = None
  if 'answer_box' in search_data:
    answer_box = format_search_results(
        search_data['answer_box'], highlight_field='answer'
    )
  else:
    answer_box = format_search_results({})
  df = pd.concat([df, pd.DataFrame([answer_box])], ignore_index=True)

  # Sort by date
  df['date'] = df['date'].apply(lambda x: format_date(x))
  df['datetime'] = pd.to_datetime(df['date'], errors='coerce')
  df = df.sort_values(by='datetime', na_position='first')
  df.replace({pd.NaT: None}, inplace=True)
  df = df.dropna(how='all')

  # Select top_k supporting evidences overall
  evidences = []

  for _, row in df.tail(num_retrieved_evidences).iterrows():
    evidences.append(
        f"""\n\nsource: {row['source']}\ndate: {row['date']}\ntitle: {row['title']}\nsnippet: {row['snippet']}\nhighlight: {row['highlight']}"""
    )
  
  return (
      ''.join(
          [
              f'\n\n\nquery: {question}',
          ]
          + evidences
      )
      + f'\n\nquestion: {question}{reasoning_and_answer}'
  )
