
from core import WeightedArgumentationFramework, AttackInferenceProblem
from categoriser import HCategoriser
from verifier import Verifier, sum_of_squares

if __name__ == '__main__':
    framework = WeightedArgumentationFramework.from_file(
        "./examples/example.apx").randomise_weights()


    categoriser = HCategoriser()

    verifier = Verifier(reducer=sum_of_squares)
    problem = AttackInferenceProblem(framework, categoriser)

    print(verifier(problem))
