"""Built-in skills that ship with Dreamland."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dreamland.memory.store import MemoryStore
    from dreamland.skills.registry import SkillRegistry

from dreamland.skills.builtin.archive_skill import ArchiveSkill
from dreamland.skills.builtin.ascii_skill import AsciiSkill
from dreamland.skills.builtin.base_convert_skill import BaseConvertSkill
from dreamland.skills.builtin.bookmark_skill import BookmarkSkill
from dreamland.skills.builtin.calendar_skill import CalendarSkill
from dreamland.skills.builtin.cert_skill import CertSkill
from dreamland.skills.builtin.changelog_gen_skill import ChangelogGenSkill
from dreamland.skills.builtin.cheat_skill import CheatSkill
from dreamland.skills.builtin.claude_skill import ClaudeSkill
from dreamland.skills.builtin.clipboard import ClipboardSkill
from dreamland.skills.builtin.clipboard_history_skill import ClipboardHistorySkill
from dreamland.skills.builtin.codex_skill import CodexSkill
from dreamland.skills.builtin.color_skill import ColorSkill
from dreamland.skills.builtin.convert_skill import ConvertSkill
from dreamland.skills.builtin.country_skill import CountrySkill
from dreamland.skills.builtin.cron_skill import CronSkill
from dreamland.skills.builtin.crontab_skill import CrontabSkill
from dreamland.skills.builtin.csv_skill import CsvSkill
from dreamland.skills.builtin.currency_skill import CurrencySkill
from dreamland.skills.builtin.cve_skill import CveSkill
from dreamland.skills.builtin.data import DataSkill
from dreamland.skills.builtin.date_format_skill import DateFormatSkill
from dreamland.skills.builtin.diagram_skill import DiagramSkill
from dreamland.skills.builtin.diff_skill import DiffSkill
from dreamland.skills.builtin.dns_skill import DnsSkill
from dreamland.skills.builtin.docker_skill import DockerSkill
from dreamland.skills.builtin.dotenv_skill import DotenvSkill
from dreamland.skills.builtin.emoji_skill import EmojiSkill
from dreamland.skills.builtin.env_skill import EnvSkill
from dreamland.skills.builtin.figlet_skill import FigletSkill
from dreamland.skills.builtin.filesystem import FileSystemSkill
from dreamland.skills.builtin.gcal_skill import GCalSkill
from dreamland.skills.builtin.git import GitSkill
from dreamland.skills.builtin.github_actions_skill import GithubActionsSkill
from dreamland.skills.builtin.github_skill import GithubSkill
from dreamland.skills.builtin.gitignore_skill import GitignoreSkill
from dreamland.skills.builtin.gmail_skill import GmailSkill
from dreamland.skills.builtin.hackernews_skill import HackerNewsSkill
from dreamland.skills.builtin.hash_skill import HashSkill
from dreamland.skills.builtin.http_header_skill import HttpHeaderSkill
from dreamland.skills.builtin.http_skill import HttpSkill
from dreamland.skills.builtin.image_skill import ImageSkill
from dreamland.skills.builtin.ip_calc_skill import IpCalcSkill
from dreamland.skills.builtin.json_skill import JsonSkill
from dreamland.skills.builtin.jwt_gen_skill import JwtGenSkill
from dreamland.skills.builtin.jwt_skill import JwtSkill
from dreamland.skills.builtin.keychain_skill import KeychainSkill
from dreamland.skills.builtin.knowledge_skill import KnowledgeSkill
from dreamland.skills.builtin.lint_skill import LintSkill
from dreamland.skills.builtin.log_analyzer_skill import LogAnalyzerSkill
from dreamland.skills.builtin.make_skill import MakeSkill
from dreamland.skills.builtin.man_skill import ManSkill
from dreamland.skills.builtin.markdown_skill import MarkdownSkill
from dreamland.skills.builtin.math_skill import MathSkill
from dreamland.skills.builtin.memory_skill import MemorySkill
from dreamland.skills.builtin.metrics_skill import MetricsSkill
from dreamland.skills.builtin.mime_skill import MimeSkill
from dreamland.skills.builtin.network import NetworkSkill
from dreamland.skills.builtin.note_skill import NoteSkill
from dreamland.skills.builtin.npm_registry_skill import NpmRegistrySkill
from dreamland.skills.builtin.npm_skill import NpmSkill
from dreamland.skills.builtin.openapi_skill import OpenApiSkill
from dreamland.skills.builtin.openrouter_skill import OpenRouterSkill
from dreamland.skills.builtin.pdf_skill import PdfSkill
from dreamland.skills.builtin.pip_skill import PipSkill
from dreamland.skills.builtin.placeholder_skill import PlaceholderSkill
from dreamland.skills.builtin.pomodoro_skill import PomodoroSkill
from dreamland.skills.builtin.port_scanner_skill import PortScannerSkill
from dreamland.skills.builtin.process_skill import ProcessSkill
from dreamland.skills.builtin.pypi_skill import PypiSkill
from dreamland.skills.builtin.qr_skill import QrSkill
from dreamland.skills.builtin.quote_skill import QuoteSkill
from dreamland.skills.builtin.random_skill import RandomSkill
from dreamland.skills.builtin.reddit_skill import RedditSkill
from dreamland.skills.builtin.regex_skill import RegexSkill
from dreamland.skills.builtin.rss_skill import RssSkill
from dreamland.skills.builtin.search import SearchSkill
from dreamland.skills.builtin.security_skill import SecuritySkill
from dreamland.skills.builtin.semver_skill import SemverSkill
from dreamland.skills.builtin.shell import ShellSkill
from dreamland.skills.builtin.snippet_gen_skill import SnippetGenSkill
from dreamland.skills.builtin.sql_skill import SqlSkill
from dreamland.skills.builtin.ssh_skill import SshSkill
from dreamland.skills.builtin.stackoverflow_skill import StackOverflowSkill
from dreamland.skills.builtin.string_skill import StringSkill
from dreamland.skills.builtin.system import SystemSkill
from dreamland.skills.builtin.systemd_skill import SystemdSkill
from dreamland.skills.builtin.template_gen_skill import TemplateGenSkill
from dreamland.skills.builtin.text_skill import TextSkill
from dreamland.skills.builtin.time_skill import TimeSkill
from dreamland.skills.builtin.todo_skill import TodoSkill
from dreamland.skills.builtin.translate_skill import TranslateSkill
from dreamland.skills.builtin.typo_skill import TypoSkill
from dreamland.skills.builtin.tz_skill import TimezoneSkill
from dreamland.skills.builtin.uptime_skill import UptimeSkill
from dreamland.skills.builtin.url_skill import UrlSkill
from dreamland.skills.builtin.uuid_skill import UuidSkill
from dreamland.skills.builtin.weather_skill import WeatherSkill
from dreamland.skills.builtin.web import WebFetchSkill
from dreamland.skills.builtin.webhook_trigger_skill import WebhookTriggerSkill
from dreamland.skills.builtin.whois_skill import WhoisSkill
from dreamland.skills.builtin.wikipedia_skill import WikipediaSkill
from dreamland.skills.builtin.xml_skill import XmlSkill
from dreamland.skills.builtin.yaml_skill import YamlSkill

__all__ = [
    "FileSystemSkill",
    "ShellSkill",
    "WebFetchSkill",
    "MemorySkill",
    "GitSkill",
    "SearchSkill",
    "ClipboardSkill",
    "DataSkill",
    "SystemSkill",
    "TimeSkill",
    "NetworkSkill",
    "HashSkill",
    "EnvSkill",
    "RegexSkill",
    "ConvertSkill",
    "JsonSkill",
    "DiffSkill",
    "ArchiveSkill",
    "CronSkill",
    "MarkdownSkill",
    "HttpSkill",
    "SqlSkill",
    "ImageSkill",
    "ProcessSkill",
    "TextSkill",
    "KnowledgeSkill",
    "TranslateSkill",
    "SecuritySkill",
    "TodoSkill",
    "TemplateGenSkill",
    "MathSkill",
    "DockerSkill",
    "CalendarSkill",
    "QrSkill",
    "JwtSkill",
    "ColorSkill",
    "UuidSkill",
    "YamlSkill",
    "SnippetGenSkill",
    "CsvSkill",
    "SemverSkill",
    "IpCalcSkill",
    "DotenvSkill",
    "LogAnalyzerSkill",
    "HttpHeaderSkill",
    "AsciiSkill",
    "StringSkill",
    "SshSkill",
    "NpmSkill",
    "PipSkill",
    "MetricsSkill",
    "PdfSkill",
    "PlaceholderSkill",
    "WebhookTriggerSkill",
    "GitignoreSkill",
    "LintSkill",
    "DiagramSkill",
    "ChangelogGenSkill",
    "NoteSkill",
    "ClipboardHistorySkill",
    "CrontabSkill",
    "BookmarkSkill",
    "KeychainSkill",
    "OpenApiSkill",
    "TypoSkill",
    "MakeSkill",
    "ManSkill",
    "GithubSkill",
    "PypiSkill",
    "CertSkill",
    "RandomSkill",
    "CountrySkill",
    "CodexSkill",
    "JwtGenSkill",
    "CheatSkill",
    "MimeSkill",
    "QuoteSkill",
    "NpmRegistrySkill",
    "CveSkill",
    "XmlSkill",
    "BaseConvertSkill",
    "PortScannerSkill",
    "TimezoneSkill",
    "RssSkill",
    "OpenRouterSkill",
    "SystemdSkill",
    "DateFormatSkill",
    "GithubActionsSkill",
    "EmojiSkill",
    "UrlSkill",
    "FigletSkill",
    "PomodoroSkill",
    "UptimeSkill",
    "WhoisSkill",
    "DnsSkill",
    "StackOverflowSkill",
    "RedditSkill",
    "CurrencySkill",
    "HackerNewsSkill",
    "WikipediaSkill",
    "WeatherSkill",
    "ClaudeSkill",
]


def register_builtins(
    registry: SkillRegistry,
    memory_store: MemoryStore | None = None,
) -> None:
    """Register all built-in skills."""

    registry.register(FileSystemSkill())
    registry.register(ShellSkill())
    registry.register(WebFetchSkill())
    registry.register(MemorySkill(store=memory_store))
    registry.register(GitSkill())
    registry.register(SearchSkill())
    registry.register(ClipboardSkill())
    registry.register(DataSkill())
    registry.register(SystemSkill())
    registry.register(TimeSkill())
    registry.register(NetworkSkill())
    registry.register(HashSkill())
    registry.register(EnvSkill())
    registry.register(RegexSkill())
    registry.register(ConvertSkill())
    registry.register(JsonSkill())
    registry.register(DiffSkill())
    registry.register(ArchiveSkill())
    registry.register(CronSkill())
    registry.register(MarkdownSkill())
    registry.register(HttpSkill())
    registry.register(SqlSkill())
    registry.register(ImageSkill())
    registry.register(ProcessSkill())
    registry.register(TextSkill())
    registry.register(KnowledgeSkill())
    registry.register(TranslateSkill())
    registry.register(SecuritySkill())
    registry.register(TodoSkill())
    registry.register(TemplateGenSkill())
    registry.register(MathSkill())
    registry.register(DockerSkill())
    registry.register(CalendarSkill())
    registry.register(QrSkill())
    registry.register(JwtSkill())
    registry.register(ColorSkill())
    registry.register(UuidSkill())
    registry.register(YamlSkill())
    registry.register(SnippetGenSkill())
    registry.register(CsvSkill())
    registry.register(SemverSkill())
    registry.register(IpCalcSkill())
    registry.register(DotenvSkill())
    registry.register(LogAnalyzerSkill())
    registry.register(HttpHeaderSkill())
    registry.register(AsciiSkill())
    registry.register(StringSkill())
    registry.register(SshSkill())
    registry.register(NpmSkill())
    registry.register(PipSkill())
    registry.register(MetricsSkill())
    registry.register(PdfSkill())
    registry.register(PlaceholderSkill())
    registry.register(WebhookTriggerSkill())
    registry.register(GitignoreSkill())
    registry.register(LintSkill())
    registry.register(DiagramSkill())
    registry.register(ChangelogGenSkill())
    registry.register(NoteSkill())
    registry.register(ClipboardHistorySkill())
    registry.register(CrontabSkill())
    registry.register(BookmarkSkill())
    registry.register(KeychainSkill())
    registry.register(OpenApiSkill())
    registry.register(TypoSkill())
    registry.register(MakeSkill())
    registry.register(ManSkill())
    registry.register(GithubSkill())
    registry.register(PypiSkill())
    registry.register(CertSkill())
    registry.register(RandomSkill())
    registry.register(CountrySkill())
    registry.register(CodexSkill())
    registry.register(JwtGenSkill())
    registry.register(CheatSkill())
    registry.register(MimeSkill())
    registry.register(QuoteSkill())
    registry.register(NpmRegistrySkill())
    registry.register(CveSkill())
    registry.register(XmlSkill())
    registry.register(BaseConvertSkill())
    registry.register(PortScannerSkill())
    registry.register(TimezoneSkill())
    registry.register(RssSkill())
    registry.register(OpenRouterSkill())
    registry.register(SystemdSkill())
    registry.register(DateFormatSkill())
    registry.register(GithubActionsSkill())
    registry.register(EmojiSkill())
    registry.register(UrlSkill())
    registry.register(FigletSkill())
    registry.register(PomodoroSkill())
    registry.register(UptimeSkill())
    registry.register(WhoisSkill())
    registry.register(DnsSkill())
    registry.register(StackOverflowSkill())
    registry.register(RedditSkill())
    registry.register(CurrencySkill())
    registry.register(HackerNewsSkill())
    registry.register(WikipediaSkill())
    registry.register(WeatherSkill())
    registry.register(ClaudeSkill())
    registry.register(GmailSkill())
    registry.register(GCalSkill())
