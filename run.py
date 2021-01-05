from run_utils import override_argv_params_from_file, get_experiment_entry_point


def main():
    entry_point = get_experiment_entry_point()
    override_argv_params_from_file()
    entry_point()


if __name__ == '__main__':
    main()
