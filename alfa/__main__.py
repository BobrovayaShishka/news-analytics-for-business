import logging

from alfa.build import build_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main() -> None:
    stats = build_all()
    print(
        f"{stats['profile_version']}: профилей {stats['profiles']['unique_clients']}, "
        f"с рёбрами {stats['companies_with_edges']}, "
        f"медиана покрытия {stats['median_known_share_both_directions']:.3f}"
    )


if __name__ == "__main__":
    main()
