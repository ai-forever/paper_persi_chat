from bs4 import BeautifulSoup
from io import StringIO, BytesIO
import re

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import HTMLConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


def convert_pdf_to_html(in_file):
    output_string = BytesIO()
    #with open(file_path, 'rb') as in_file:
    parser = PDFParser(in_file)
    doc = PDFDocument(parser)
    rsrcmgr = PDFResourceManager()
    device = HTMLConverter(rsrcmgr, output_string, codec='utf-8', laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    for page in PDFPage.create_pages(doc):
        interpreter.process_page(page)

    return(output_string.getvalue())


def text_postproc(text):
    return re.sub(r'\s+', ' ', text.strip().replace('-\n', '').replace('- ', ''))


def sci_parse_pdf_article(file_path, min_span_len=5, min_seg_words=1):
    parsed_article = convert_pdf_to_html(file_path)
    parsed_article = BeautifulSoup(parsed_article, "html")
    article_spans = parsed_article.find_all('span')
    span_fonts = []
    for span in article_spans:
        if len(span.find_all('span')) == 0:
            fonts = re.findall('font-size:(\d+)px', span.attrs['style'])
            if len(fonts) > 0:
                font = int(fonts[0])
                span_fonts.append([font, span.text])

    all_fonts = [el[0] for el in span_fonts]
    main_font = {}
    for f, t in span_fonts:
        main_font[f] = main_font.get(f, 0) + len(t)
    main_font = max(main_font, key=main_font.get)

    # fonts > main_font => header
    for sp in span_fonts:
        sp[0] = main_font if sp[0] <= main_font else sp[0]

    # join segments by font
    joined_segments = []
    cur_font = ''
    cur_segment = []
    for sp in span_fonts:
        # each header should end by \n
        if sp[0] == cur_font or (len(cur_segment) > 0 and not cur_segment[-1].endswith('\n')):
            cur_segment.append(sp[1])
        else:
            if len(cur_segment) > 0:
                joined_segments.append((cur_font,  text_postproc(''.join(cur_segment))))
            cur_font, cur_segment = sp[0], [sp[1]]
    if len(cur_segment) > 0:
        joined_segments.append((cur_font, text_postproc(''.join(cur_segment))))

    # filter short & select title
    joined_segments = [seg for seg in joined_segments if len(seg[1]) > min_span_len and\
                               sum(1 for w in seg[1].split() if len(w) >= 3) >= min_seg_words]
    try:
        title = [j for j in joined_segments if j[0] > main_font][0][1]
    except:
        title = ""
    for i in range(len(joined_segments)):
        if joined_segments[i][1] == title:
            joined_segments = joined_segments[i+1:]
            break

    sections = []
    seg_title = "section"
    for font, seg in joined_segments:
        if font == main_font:
            if "reference" not in seg_title.lower():
                if seg_title == 'section' and len(sections) > 0:
                    sections[-1]['text'] += ' ' + seg
                else:
                    sections.append({'title': seg_title, 'text': seg})
            seg_title = "section"
        else:
            seg_title = seg
            
    # section types by keywords
    for i, section in enumerate(sections):
        title_low = section['title'].lower()
        if 'abstract' in title_low:
            section['title'] = 'Abstract'
            section['type'] = 'abstract'
        elif 'introduction' in title_low:
            section['title'] = 'Introduction'
            section['type'] = 'introduction'
        elif 'acknowledgments' in title_low or 'acknowledgements' in title_low:
            section['type'] = 'acknowledgments'
        elif 'results' in title_low or 'experiments' in title_low:
            section['type'] = 'experiments'
        elif 'conclusion' in title_low:
            section['type'] = 'conclusion'
        elif 'related' in title_low:
            section['type'] = 'related work'
        elif i == 0:
            section['type'] = 'abstract'
        else:
            section['type'] = 'method'
    
    abstract_sections = [i for i in range(len(sections)) if sections[i]['type'] == 'abstract']
    sections = sections[abstract_sections[-1]:]
    
    segments = []
    for i, el in enumerate(sections):
        segments.append([str(i), el['text'], el['title'], el['type']])
    result = {'title': title, 'id': 'userid', 'segments': segments}
    return result
