import unittest

from deeptempest_ubuntu.cli import build_parser
from deeptempest_ubuntu.core import check_project_layout


class TestUbuntuCli(unittest.TestCase):
    def test_parser_accepts_doctor(self):
        parser = build_parser()
        args = parser.parse_args(["doctor"])
        self.assertEqual(args.command, "doctor")

    def test_layout_check_has_items(self):
        report = check_project_layout(["README.md"])
        self.assertTrue(report.items)
        self.assertTrue(report.items[0].ok)


if __name__ == "__main__":
    unittest.main()
