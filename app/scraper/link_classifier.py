"""
link_classifier.py
------------------
Classifies external links found inside Google Docs and returns
enrichment metadata written inline into the markdown output.

Called by extractor.py for every link that is NOT a Google Doc.
The enrichment tells a RAG system what a linked resource contains
and when to refer a student to it — without ever scraping that resource.

Functions in this file:
    classify_external_link()        — Public. Classify one URL, return ExternalLinkInfo.
    format_link_as_markdown_note()  — Public. Format classification as a markdown block.
    _classify_google_sheets()       — Internal. Handle spreadsheet links.
    _classify_google_forms()        — Internal. Handle form links.
    _classify_google_presentation() — Internal. Handle slide deck links.
    _classify_youtube()             — Internal. Handle video links.
    _classify_kaggle()              — Internal. Handle Kaggle competition links.
    _classify_github()              — Internal. Handle GitHub repo links.
    _classify_iitm_portals()        — Internal. Handle official IITM site links.
    _classify_nptel()               — Internal. Handle NPTEL course links.
    _classify_other_known_sites()   — Internal. Handle other known domains.
    _generic_fallback_classification() — Internal. Fallback for unknown links.
    _identify_course_name_from_context() — Internal. Detect course from context text.
"""

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

logger = logging.getLogger("scraper.link_classifier")


# =============================================================================
# DATA STRUCTURE
# =============================================================================

@dataclass
class ExternalLinkInfo:
    """
    Holds enrichment data for one external link.

    Attributes:
        title:            Short human-readable name for this resource.
        what_it_contains: What content or data this linked page holds.
        when_to_refer:    When a student should use this link.
        category:         Machine-readable tag, e.g. "form", "sheet", "video".
        access:           "public" or "restricted".
    """
    title: str
    what_it_contains: str
    when_to_refer: str
    category: str
    access: str = "public"


# =============================================================================
# PUBLIC
# =============================================================================

def classify_external_link(
    url: str,
    section_heading: str,
    surrounding_paragraph_text: str,
    parent_doc_title: str,
) -> ExternalLinkInfo:
    """
    Classify an external link and return enrichment metadata.

    Args:
        url: The full external URL to classify.
        section_heading: The heading of the section where this link appears.
        surrounding_paragraph_text: The paragraph or list item containing the link.
        parent_doc_title: Title of the Google Doc where this link was found.

    Returns:
        ExternalLinkInfo with title, description, and category filled in.
        Always returns a result — uses generic fallback for unknown links.
    """
    url_lowercase = url.lower()
    combined_context = f"{section_heading.lower()} {surrounding_paragraph_text.lower()}"

    classification = (
        _classify_google_sheets(url, url_lowercase, combined_context)
        or _classify_google_forms(url, url_lowercase, combined_context)
        or _classify_google_presentation(url, url_lowercase, combined_context)
        or _classify_youtube(url, url_lowercase, combined_context)
        or _classify_kaggle(url, url_lowercase, combined_context)
        or _classify_github(url, url_lowercase, combined_context)
        or _classify_iitm_portals(url, url_lowercase, combined_context)
        or _classify_nptel(url, url_lowercase, combined_context)
        or _classify_other_known_sites(url, url_lowercase, combined_context)
        or _generic_fallback_classification(url, section_heading, parent_doc_title)
    )

    logger.debug(f"Classified as [{classification.category}]: {url[:60]}")
    return classification


def format_link_as_markdown_note(url: str, link_info: ExternalLinkInfo) -> str:
    """
    Format an ExternalLinkInfo as a blockquote reference note in markdown.

    Args:
        url: The original URL.
        link_info: Classification result from classify_external_link().

    Returns:
        A multi-line markdown blockquote string ready to insert into a .md file.
    """
    return (
        f"\n> 🔗 **External Reference**\n"
        f"> **Title:** {link_info.title}\n"
        f"> **What it contains:** {link_info.what_it_contains}\n"
        f"> **When to refer:** {link_info.when_to_refer}\n"
        f"> **Category:** `{link_info.category}`\n"
        f"> **Access:** {link_info.access}\n"
        f"> **URL:** {url}\n"
    )


# =============================================================================
# INTERNAL CLASSIFIERS
# Each returns ExternalLinkInfo if the URL matches, or None if it does not.
# classify_external_link() tries them in order and uses the first match.
# =============================================================================

def _classify_google_sheets(
    url: str, url_lowercase: str, combined_context: str
) -> ExternalLinkInfo | None:
    """Classify Google Sheets spreadsheet links."""

    if "spreadsheets" not in url_lowercase:
        return None

    if "nptel" in combined_context or "swayam" in combined_context:
        return ExternalLinkInfo(
            title="NPTEL/SWAYAM Courses Eligible for Credit Transfer",
            what_it_contains="List of NPTEL/SWAYAM courses that count toward IITM BS degree credits",
            when_to_refer="Student asks about NPTEL credit transfer or SWAYAM electives",
            category="course_list",
        )
    if "exam city" in combined_context or "exam centre" in combined_context:
        return ExternalLinkInfo(
            title="Exam City Details Sheet",
            what_it_contains="List of cities where IITM BS in-person exams are held",
            when_to_refer="Student asks about exam cities or centre locations",
            category="exam_info",
        )
    if "bdm" in combined_context and "project" in combined_context:
        return ExternalLinkInfo(
            title="BDM Project Submission Timeline",
            what_it_contains="Deadlines and timeline for BDM project submission",
            when_to_refer="Student asks about BDM project deadline",
            category="project_timeline",
        )
    if "viva" in combined_context and "eligib" in combined_context:
        return ExternalLinkInfo(
            title="MLP Viva Eligibility Sheet",
            what_it_contains="Sheet showing which students are eligible for the MLP viva",
            when_to_refer="Student asks about MLP viva eligibility",
            category="project_status",
        )
    if "registration status" in combined_context:
        return ExternalLinkInfo(
            title="MLP Project Registration Status Sheet",
            what_it_contains="Sheet showing MLP project registration confirmation per student",
            when_to_refer="Student asks about MLP registration status",
            category="project_status",
        )
    if "orientation" in combined_context:
        return ExternalLinkInfo(
            title="Course-wise Orientation Video Links",
            what_it_contains="YouTube links for orientation videos for each course at all levels",
            when_to_refer="Student asks about a course orientation or introduction video",
            category="orientation",
        )
    if "region" in combined_context or "house" in combined_context:
        return ExternalLinkInfo(
            title="IITM BS Region and House Reference Sheet",
            what_it_contains="Mapping of exam cities to IITM BS regions and student houses",
            when_to_refer="Student asks about their region, house, or exam city mapping",
            category="exam_info",
        )
    if "degree level" in combined_context:
        return ExternalLinkInfo(
            title="Degree Level Courses List",
            what_it_contains="Complete list of courses available at the BS degree level",
            when_to_refer="Student asks about degree level courses",
            category="course_list",
        )
    if "masters" in combined_context or "research" in combined_context:
        return ExternalLinkInfo(
            title="Masters and Research Program Reference Sheet",
            what_it_contains="Reference data for pathways to Masters or Research programs",
            when_to_refer="Student asks about Masters admission, research program, or MTech pathways",
            category="pathway",
        )

    return ExternalLinkInfo(
        title="Reference Spreadsheet",
        what_it_contains=f"Google Sheet referenced in context: {combined_context[:80]}",
        when_to_refer="Student asks about data in this spreadsheet",
        category="reference_sheet",
    )


def _classify_google_forms(
    url: str, url_lowercase: str, combined_context: str
) -> ExternalLinkInfo | None:
    """Classify Google Forms links."""

    is_google_form = "forms.gle" in url_lowercase or "docs.google.com/forms" in url_lowercase
    if not is_google_form:
        return None

    course_name = _identify_course_name_from_context(combined_context)

    if "registration" in combined_context:
        return ExternalLinkInfo(
            title=f"{course_name} Project Registration Form",
            what_it_contains=f"Google Form to register for the {course_name} project",
            when_to_refer=f"Student asks about {course_name} project registration",
            category="registration_form",
        )
    if "feedback" in combined_context or "survey" in combined_context:
        return ExternalLinkInfo(
            title=f"{course_name} Feedback Form",
            what_it_contains=f"Feedback or survey form for {course_name}",
            when_to_refer=f"Student needs to submit feedback for {course_name}",
            category="feedback_form",
        )

    return ExternalLinkInfo(
        title=f"{course_name} Form",
        what_it_contains=f"Google Form for {course_name}",
        when_to_refer=f"Student needs to fill out a form for {course_name}",
        category="form",
    )


def _classify_google_presentation(
    url: str, url_lowercase: str, combined_context: str
) -> ExternalLinkInfo | None:
    """Classify Google Slides presentation links."""

    if "presentation" not in url_lowercase:
        return None

    if "pgd" in combined_context or "mtech" in combined_context:
        return ExternalLinkInfo(
            title="PGD and MTech Upgrade Pathway Presentation",
            what_it_contains="Slides introducing the PGD and MTech upgrade option for BS students",
            when_to_refer="Student asks about PGD, MTech upgrade, or postgraduate diploma",
            category="presentation",
        )
    if "dl" in combined_context and "genai" in combined_context:
        return ExternalLinkInfo(
            title="DL GenAI Project Registration Process Slides",
            what_it_contains="Step-by-step slides for DL GenAI project registration",
            when_to_refer="Student asks about DL GenAI project registration process",
            category="project_guide",
        )

    return ExternalLinkInfo(
        title="Reference Presentation",
        what_it_contains=f"Google Slides presentation in context: {combined_context[:80]}",
        when_to_refer="Student asks about the topic in this presentation",
        category="presentation",
    )


def _classify_youtube(
    url: str, url_lowercase: str, combined_context: str
) -> ExternalLinkInfo | None:
    """Classify YouTube video links."""

    if "youtube.com" not in url_lowercase and "youtu.be" not in url_lowercase:
        return None

    if "oppe" in combined_context or "camera" in combined_context:
        return ExternalLinkInfo(
            title="OPPE Mobile Camera Setup Tutorial",
            what_it_contains="Video showing how to position mobile camera during OPPE and SCT exams",
            when_to_refer="Student asks about OPPE camera setup or SCT exam camera requirements",
            category="tutorial_video",
        )
    if "submit" in combined_context or "submission" in combined_context:
        return ExternalLinkInfo(
            title="Project Submission Guide Video",
            what_it_contains="Video guide for submitting a project including the viva workflow",
            when_to_refer="Student asks how to submit a project",
            category="tutorial_video",
        )
    if "notebook" in combined_context or "kaggle" in combined_context:
        return ExternalLinkInfo(
            title="Kaggle Notebook Submission Demo",
            what_it_contains="Demo video for creating and submitting a Kaggle notebook",
            when_to_refer="Student asks about Kaggle notebook submission",
            category="tutorial_video",
        )

    return ExternalLinkInfo(
        title="Reference Video",
        what_it_contains=f"YouTube video in context: {combined_context[:80]}",
        when_to_refer="Student asks about the topic in this video",
        category="video",
    )


def _classify_kaggle(
    url: str, url_lowercase: str, combined_context: str
) -> ExternalLinkInfo | None:
    """Classify Kaggle competition and platform links."""

    if "kaggle.com" not in url_lowercase:
        return None

    course_name = _identify_course_name_from_context(combined_context)

    if "/t/" in url_lowercase or "competition" in url_lowercase:
        return ExternalLinkInfo(
            title=f"Kaggle Competition — {course_name} Project",
            what_it_contains=f"Kaggle competition page for the {course_name} project submission",
            when_to_refer=f"Student asks about the {course_name} Kaggle competition link",
            category="project_submission",
        )

    return ExternalLinkInfo(
        title="Kaggle Platform",
        what_it_contains="Kaggle platform for ML competitions and notebook submissions",
        when_to_refer="Student asks about Kaggle or needs the Kaggle platform link",
        category="external_tool",
    )


def _classify_github(
    url: str, url_lowercase: str, combined_context: str
) -> ExternalLinkInfo | None:
    """Classify GitHub repository links."""

    if "github.com" not in url_lowercase:
        return None

    if "sample" in url_lowercase or "template" in url_lowercase or "dl-genai" in url_lowercase:
        return ExternalLinkInfo(
            title="DL GenAI Project Sample Repository",
            what_it_contains="Sample GitHub repository structure and guidelines for the DL GenAI project",
            when_to_refer="Student asks about the DL GenAI GitHub repo or sample structure",
            category="project_guide",
        )

    return ExternalLinkInfo(
        title="GitHub Repository",
        what_it_contains=f"GitHub repository in context: {combined_context[:80]}",
        when_to_refer="Student asks about this GitHub repository",
        category="external_tool",
    )


def _classify_iitm_portals(
    url: str, url_lowercase: str, combined_context: str
) -> ExternalLinkInfo | None:
    """Classify official IITM and related portal links."""

    if "study.iitm.ac.in" in url_lowercase:
        return ExternalLinkInfo(
            title="IITM BS Study Portal",
            what_it_contains="Official study portal with course content, exam details, and academics",
            when_to_refer="Student asks about the study portal or qualifier exam information",
            category="official_portal",
        )
    if "onlinedegree.iitm.ac.in" in url_lowercase:
        if "privacy" in url_lowercase:
            return ExternalLinkInfo(
                title="IITM BS Privacy Policy",
                what_it_contains="Official privacy policy for the IITM BS online degree program",
                when_to_refer="Student asks about privacy policy or data collection",
                category="policy",
            )
        return ExternalLinkInfo(
            title="IITM Online Degree Portal",
            what_it_contains="Official IITM online degree portal for the BS program",
            when_to_refer="Student asks about the official IITM online degree website",
            category="official_portal",
        )
    if "iitmaa.org" in url_lowercase:
        return ExternalLinkInfo(
            title="IITM Alumni Association (IITMAA)",
            what_it_contains="Alumni portal. One-time registration fee of Rs 7080 for alumni benefits.",
            when_to_refer="Student asks about alumni benefits, IITMAA, or alumni registration",
            category="official_portal",
        )
    if "research.iitm.ac.in" in url_lowercase:
        return ExternalLinkInfo(
            title="IITM Research Portal",
            what_it_contains="Portal with details on Masters and PhD research programs at IITM",
            when_to_refer="Student asks about IITM research programs or PhD admission",
            category="official_portal",
        )
    if "tds.s-anand.net" in url_lowercase:
        return ExternalLinkInfo(
            title="Tools in Data Science (TDS) Course Portal",
            what_it_contains="Official TDS portal with assignments, projects, and ROE links. Seek Portal not used for TDS.",
            when_to_refer="Student asks about TDS course content, assignments, or the TDS portal",
            category="course_portal",
        )
    if "lookerstudio.google.com" in url_lowercase:
        return ExternalLinkInfo(
            title="DL GenAI Registration Status Dashboard",
            what_it_contains="Looker Studio dashboard showing DL GenAI project registration status",
            when_to_refer="Student asks about DL GenAI registration confirmation",
            category="project_status",
        )

    return None


def _classify_nptel(
    url: str, url_lowercase: str, combined_context: str
) -> ExternalLinkInfo | None:
    """Classify NPTEL course links."""

    if "nptel.ac.in" not in url_lowercase:
        return None

    return ExternalLinkInfo(
        title=f"NPTEL Course — {combined_context[:60]}",
        what_it_contains=f"NPTEL course page: {combined_context[:100]}",
        when_to_refer="Student asks about this NPTEL course or its credit transfer eligibility",
        category="nptel_course",
    )


def _classify_other_known_sites(
    url: str, url_lowercase: str, combined_context: str
) -> ExternalLinkInfo | None:
    """Classify other well-known domains that appear in IITM docs."""

    if "exam.sanand.workers.dev" in url_lowercase:
        return ExternalLinkInfo(
            title="TDS Entrance Exam",
            what_it_contains="Prerequisite exam students must pass before registering for TDS",
            when_to_refer="Student asks about TDS entrance exam or TDS prerequisites",
            category="exam",
        )

    # IITM student house portals — named after Indian wildlife sanctuaries.
    house_domain_keywords = [
        "iitmbs.org", "bandipur", "corbett", "kanha", "kaziranga",
        "nallamala", "namdapha", "nilgiri", "pichavaram", "saranda",
        "sundarbans", "wayanad", "gir.",
    ]
    if any(keyword in url_lowercase for keyword in house_domain_keywords):
        parsed_url = urlparse(url)
        house_name = parsed_url.hostname.split(".")[0].title() if parsed_url.hostname else "Student House"
        return ExternalLinkInfo(
            title=f"{house_name} House — IITM BS",
            what_it_contains=f"Portal for {house_name} House in the IITM BS student house system",
            when_to_refer=f"Student asks about {house_name} House or their student house portal",
            category="house_portal",
        )

    return None


def _generic_fallback_classification(
    url: str,
    section_heading: str,
    parent_doc_title: str,
) -> ExternalLinkInfo:
    """
    Fallback for any link that did not match any known pattern.
    Always returns an ExternalLinkInfo — never returns None.

    Args:
        url: The external URL.
        section_heading: Section heading where the link was found.
        parent_doc_title: Title of the parent Google Doc.

    Returns:
        Generic ExternalLinkInfo based on domain name.
    """
    domain = urlparse(url).netloc or url[:40]
    return ExternalLinkInfo(
        title=f"External Resource — {domain}",
        what_it_contains=f"External link from '{parent_doc_title}', section: '{section_heading}'",
        when_to_refer=f"Student asks about {section_heading[:60]}",
        category="external_reference",
    )


# =============================================================================
# CONTEXT HELPER
# =============================================================================

def _identify_course_name_from_context(combined_context: str) -> str:
    """
    Identify which IITM BS course a link relates to from surrounding text.

    Args:
        combined_context: Section heading + paragraph text, both lowercased.

    Returns:
        Short course name string. Returns "the course" if no match found.
    """
    if "mlp" in combined_context or "machine learning practice" in combined_context:
        return "MLP"
    if "dl" in combined_context and "genai" in combined_context:
        return "DL GenAI"
    if "mad 1" in combined_context or "mad-1" in combined_context or "app dev 1" in combined_context:
        return "MAD-1"
    if "mad 2" in combined_context or "mad-2" in combined_context or "app dev 2" in combined_context:
        return "MAD-2"
    if "bdm" in combined_context or "business data management" in combined_context:
        return "BDM"
    if "tds" in combined_context or "tools in data science" in combined_context:
        return "TDS"
    return "the course"
