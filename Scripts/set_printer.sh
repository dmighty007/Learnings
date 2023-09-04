sudo pacman -S cups
sudo systemctl enable --now cups
sudo systemctl start --now cups
sudo pacman -S hplip
sudo hp-setup -i
yay -S system-config-printer
