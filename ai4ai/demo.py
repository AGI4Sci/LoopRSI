from .contracts import ResearchState, SkillContext
from .research_loop import ResearchLoop
from .router import SkillRouter
from skills.example import EchoSkill


def main() -> None:
    context = SkillContext(ResearchState(task_id="dummy", question="Can a skill inspect research state?"))
    router = SkillRouter([EchoSkill()])
    loop = ResearchLoop(router)
    print(loop.invoke("example.echo", context))


if __name__ == "__main__":
    main()

