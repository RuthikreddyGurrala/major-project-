let myFormEl = document.getElementById("myForm");

let nameEl = document.getElementById("name");
let nameErrMsgEl = document.getElementById("nameErrMsg");

let emailEl = document.getElementById("email");
let emailErrMsgEl = document.getElementById("emailErrMsg");

let button1 = document.getElementById("button1");
nameEl.addEventListener("blur", function(event) {
    if (event.target.value === "") {
        nameErrMsgEl.textContent = "Required*";
    } else {
        nameErrMsgEl.textContent = "";
    }
});

emailEl.addEventListener("blur", function(event) {
    if (event.target.value === "") {
        emailErrMsgEl.textContent = "Required*";
    } else {
        emailErrMsgEl.textContent = "";
    }
});

myFormEl.addEventListener("submit", function(event) {
    if (nameEl.value === "ruthikreddygurrala@gmail.com" && emailEl.value === "123456") {
        button1.textContent = "Login Success";
    } else {
        button1.textContent = "Incorrect Login Credintials";
    }
    event.preventDefault();
});