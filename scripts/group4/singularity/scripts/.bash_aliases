train_openpi() {
    cd $HOME/workspace/hsr_openpi
    bash $HOME/workspace/hsr_openpi/scripts_launch/train_openpi.sh || {
        echo "Failed to execute train_openpi.sh"
        exit 1
    }
}
alias train=train_openpi

