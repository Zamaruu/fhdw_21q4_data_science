function navbarClick() {
    var x = document.getElementById("myTopnav");
    var icon = document.getElementById("topnav-icon")
    if (x.className === "topnav") {
        x.className += " responsive";
        icon.className = "ms-Icon ms-Icon--Cancel";
    } else {
        x.className = "topnav";
        icon.className = "ms-Icon ms-Icon--GlobalNavButton";
    }
}