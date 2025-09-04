import click


@click.group()
def main():
    """Re:Mind CLI"""
    pass


@main.command()
def test():
    from _playground.pg_250826 import main5
    main5()


if __name__ == '__main__':
    main()
