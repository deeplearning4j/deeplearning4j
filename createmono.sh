shopt -s dotglob # to be able to mv all hidden files easily

mkdir deeplearning4j
git mv -k * deeplearning4j
git commit -a -m "Move deeplearning4j"

PROJECTS=(libnd4j nd4j datavec arbiter nd4s gym-java-client rl4j scalnet jumpy)
for PROJECT in ${PROJECTS[@]}; do
    git branch -D $PROJECT || true
    git checkout --orphan $PROJECT
    git reset --hard
    rm -Rf $PROJECT
    git pull https://github.com/deeplearning4j/$PROJECT master
    mkdir $PROJECT
    git mv -k * $PROJECT
    git commit -a -m "Move $PROJECT"
    git checkout feature/monorepo
    git merge --allow-unrelated-histories $PROJECT -m "Merge $PROJECT"
done

