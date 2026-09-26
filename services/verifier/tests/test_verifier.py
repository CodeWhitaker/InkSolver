import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.main import SolveStep, VerifyRequest, verify_payload


class VerifierTests(unittest.TestCase):
    def test_power_rule_integral_verifies(self):
        response = verify_payload(
            VerifyRequest(
                problem_text="Evaluate the indefinite integral of x squared.",
                subject="math",
                final_answer="\\frac{x^3}{3}+C",
                steps=[
                    SolveStep(step_num=1, latex="\\int x^2\\,dx"),
                    SolveStep(step_num=2, latex="\\frac{x^{2+1}}{2+1}+C"),
                    SolveStep(step_num=3, latex="\\frac{x^3}{3}+C"),
                ],
            ),
        )

        self.assertEqual(response.verification_status, "verified")
        self.assertTrue(response.verified)
        self.assertTrue(all(step.verification_status == "verified" for step in response.steps))

    def test_wrong_integral_mismatches(self):
        response = verify_payload(
            VerifyRequest(
                problem_text="Integrate x^2 dx.",
                subject="math",
                final_answer="x^3+C",
                steps=[SolveStep(step_num=1, latex="x^3+C")],
            ),
        )

        self.assertEqual(response.verification_status, "mismatch")
        self.assertFalse(response.verified)
        self.assertEqual(response.steps[0].verification_status, "mismatch")

    def test_derivative_verifies(self):
        response = verify_payload(
            VerifyRequest(
                problem_text="Differentiate x^3 with respect to x.",
                subject="math",
                final_answer="3x^2",
                steps=[SolveStep(step_num=1, latex="3x^2")],
            ),
        )

        self.assertEqual(response.verification_status, "verified")
        self.assertTrue(response.verified)

    def test_trig_derivative_verifies(self):
        response = verify_payload(
            VerifyRequest(
                problem_text="d/dx(sin x)",
                subject="math",
                final_answer="cos x",
                steps=[SolveStep(step_num=1, latex="\\cos x")],
            ),
        )

        self.assertEqual(response.verification_status, "verified")
        self.assertTrue(response.verified)
        self.assertEqual(response.computed_value, "cos(x)")

    def test_trig_derivative_mismatches(self):
        response = verify_payload(
            VerifyRequest(
                problem_text="\\frac{d}{dx}\\sin x",
                subject="math",
                final_answer="-\\cos x",
                steps=[SolveStep(step_num=1, latex="-\\cos x")],
            ),
        )

        self.assertEqual(response.verification_status, "mismatch")
        self.assertFalse(response.verified)
        self.assertEqual(response.computed_value, "cos(x)")

    def test_unsupported_problem_is_unverifiable(self):
        response = verify_payload(
            VerifyRequest(
                problem_text="Balance this chemical equation.",
                subject="chem",
                final_answer="2H_2 + O_2 -> 2H_2O",
                steps=[SolveStep(step_num=1, latex="2H_2 + O_2 -> 2H_2O")],
            ),
        )

        self.assertEqual(response.verification_status, "unverifiable")

    def test_limit_verifies(self):
        response = verify_payload(
            VerifyRequest(
                problem_text="\\lim_{x \\to 0} \\frac{\\sin x}{x}",
                subject="math",
                final_answer="1",
                steps=[SolveStep(step_num=1, latex="1")],
            ),
        )

        self.assertEqual(response.verification_status, "verified")
        self.assertTrue(response.verified)
        self.assertEqual(
            response.steps[0].reason,
            "This step matches the expected symbolic expression.",
        )

    def test_limit_mismatches(self):
        response = verify_payload(
            VerifyRequest(
                problem_text="\\lim_{x \\to 0} \\frac{\\sin x}{x}",
                subject="math",
                final_answer="0",
                steps=[SolveStep(step_num=1, latex="0")],
            ),
        )

        self.assertEqual(response.verification_status, "mismatch")
        self.assertFalse(response.verified)

    def test_limit_english_verifies(self):
        response = verify_payload(
            VerifyRequest(
                problem_text="limit of x^2 as x approaches 3",
                subject="math",
                final_answer="9",
                steps=[SolveStep(step_num=1, latex="9")],
            ),
        )

        self.assertEqual(response.verification_status, "verified")
        self.assertTrue(response.verified)

    def test_definite_integral_verifies(self):
        response = verify_payload(
            VerifyRequest(
                problem_text="\\int_{0}^{1} x^2 dx",
                subject="math",
                final_answer="\\frac{1}{3}",
                steps=[SolveStep(step_num=1, latex="\\frac{x^3}{3}\\Big|_0^1")],
            ),
        )

        self.assertEqual(response.verification_status, "verified")
        self.assertTrue(response.verified)
        self.assertEqual(response.computed_value, "1/3")

    def test_definite_integral_mismatches(self):
        response = verify_payload(
            VerifyRequest(
                problem_text="\\int_{0}^{1} x^2 dx",
                subject="math",
                final_answer="1",
                steps=[SolveStep(step_num=1, latex="1")],
            ),
        )

        self.assertEqual(response.verification_status, "mismatch")
        self.assertFalse(response.verified)

    def test_linear_system_verifies(self):
        response = verify_payload(
            VerifyRequest(
                problem_text="Solve the system of equations: x + y = 5 and 2x - y = 1",
                subject="math",
                final_answer="x = 2, y = 3",
                steps=[SolveStep(step_num=1, latex="x=2, y=3")],
            ),
        )

        self.assertEqual(response.verification_status, "verified")
        self.assertTrue(response.verified)

    def test_linear_system_mismatches(self):
        response = verify_payload(
            VerifyRequest(
                problem_text="Solve the system of equations: x + y = 5 and 2x - y = 1",
                subject="math",
                final_answer="x = 3, y = 2",
                steps=[SolveStep(step_num=1, latex="x=3, y=2")],
            ),
        )

        self.assertEqual(response.verification_status, "mismatch")
        self.assertFalse(response.verified)

    def test_hyperbolic_derivative_verifies(self):
        # Regression: "sinh x" was mangled to "sin(h) x" by function-call
        # normalization, so a correct "cosh x" answer mismatched.
        response = verify_payload(
            VerifyRequest(
                problem_text="Differentiate sinh x with respect to x.",
                subject="math",
                final_answer="cosh x",
                steps=[SolveStep(step_num=1, latex="\\cosh x")],
            ),
        )

        self.assertEqual(response.verification_status, "verified")
        self.assertTrue(response.verified)
        self.assertEqual(response.computed_value, "cosh(x)")

    def test_hyperbolic_derivative_mismatches(self):
        response = verify_payload(
            VerifyRequest(
                problem_text="Differentiate cosh x with respect to x.",
                subject="math",
                final_answer="cosh x",
                steps=[SolveStep(step_num=1, latex="\\cosh x")],
            ),
        )

        self.assertEqual(response.verification_status, "mismatch")
        self.assertFalse(response.verified)
        self.assertEqual(response.computed_value, "sinh(x)")

    def test_limit_to_infinity_verifies(self):
        # Regression: the point group captured just "\" for "\to\infty",
        # leaving the problem unverifiable instead of verified.
        response = verify_payload(
            VerifyRequest(
                problem_text="\\lim_{x\\to\\infty} \\frac{1}{x}",
                subject="math",
                final_answer="0",
                steps=[SolveStep(step_num=1, latex="0")],
            ),
        )

        self.assertEqual(response.verification_status, "verified")
        self.assertTrue(response.verified)
        self.assertEqual(response.computed_value, "0")

    def test_limit_other_variable_verifies(self):
        # Regression: variables outside x/y/z/t fell back to X, so the limit
        # was computed in the wrong variable and correct answers mismatched.
        response = verify_payload(
            VerifyRequest(
                problem_text="\\lim_{n \\to 2} n^2",
                subject="math",
                final_answer="4",
                steps=[SolveStep(step_num=1, latex="4")],
            ),
        )

        self.assertEqual(response.verification_status, "verified")
        self.assertTrue(response.verified)
        self.assertEqual(response.computed_value, "4")

    def test_tuple_final_answer_does_not_crash(self):
        # Regression: a tuple-shaped parse result (e.g. "2, 3") raised
        # TypeError inside simplify() and would surface as a 500.
        response = verify_payload(
            VerifyRequest(
                problem_text="Solve the system of equations: x + y = 5 and 2x - y = 1",
                subject="math",
                final_answer="2, 3",
                steps=[SolveStep(step_num=1, latex="x=2, y=3")],
            ),
        )

        self.assertEqual(response.verification_status, "mismatch")
        self.assertFalse(response.verified)

    def test_tuple_step_does_not_crash(self):
        response = verify_payload(
            VerifyRequest(
                problem_text="Integrate x^2 dx.",
                subject="math",
                final_answer="x^3/3+C",
                steps=[SolveStep(step_num=1, latex="x^2, x^3")],
            ),
        )

        self.assertEqual(response.verification_status, "verified")
        self.assertTrue(response.verified)
        self.assertEqual(response.steps[0].verification_status, "unverifiable")

    def test_linear_system_ignores_longer_words(self):
        # Regression: "max = 5" matched the "x = <val>" pattern inside "max",
        # falsely verifying an answer that never states x = 5.
        response = verify_payload(
            VerifyRequest(
                problem_text="Solve the system of equations: x + y = 6 and x - y = 4",
                subject="math",
                final_answer="the max = 5, y = 1",
                steps=[SolveStep(step_num=1, latex="x=5, y=1")],
            ),
        )

        self.assertEqual(response.verification_status, "mismatch")
        self.assertFalse(response.verified)

    def test_definite_integral_unbraced_bounds_verify(self):
        # Regression: bound extraction must handle \int_0^1 as well as
        # \int_{0}^{1}.
        response = verify_payload(
            VerifyRequest(
                problem_text="\\int_0^1 x^2 dx",
                subject="math",
                final_answer="1/3",
                steps=[SolveStep(step_num=1, latex="1/3")],
            ),
        )

        self.assertEqual(response.verification_status, "verified")
        self.assertTrue(response.verified)

    def test_definite_integral_infinite_bound_verifies(self):
        # Regression: "\infty" bounds were split into garbage by a lazy
        # capture group, and bare "e" parsed as a symbol instead of Euler.
        response = verify_payload(
            VerifyRequest(
                problem_text="\\int_{0}^{\\infty} e^{-x} dx",
                subject="math",
                final_answer="1",
                steps=[SolveStep(step_num=1, latex="1")],
            ),
        )

        self.assertEqual(response.verification_status, "verified")
        self.assertTrue(response.verified)
        self.assertEqual(response.computed_value, "1")

    def test_euler_number_parses(self):
        response = verify_payload(
            VerifyRequest(
                problem_text="Differentiate e^x with respect to x.",
                subject="math",
                final_answer="e^x",
                steps=[SolveStep(step_num=1, latex="e^x")],
            ),
        )

        self.assertEqual(response.verification_status, "verified")
        self.assertTrue(response.verified)


if __name__ == "__main__":
    unittest.main()
