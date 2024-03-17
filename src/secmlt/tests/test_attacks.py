def run_attack(attack, model, data_loader):
    return attack(model, data_loader)


def test_native_pgd_attack(native_pgd_attack, model, data_loader) -> None:
    run_attack(native_pgd_attack, model, data_loader)


def test_foolbox_pgd_attack(foolbox_pgd_attack, model, data_loader) -> None:
    run_attack(foolbox_pgd_attack, model, data_loader)
